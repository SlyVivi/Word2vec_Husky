#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <functional>
#include <cmath>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <gperftools/profiler.h>

#include "boost/utility/string_ref.hpp"
#include "boost/tokenizer.hpp"
#include "mongo/bson/bson.h"
#include "mongo/client/dbclient.h"

#include "base/serialization.hpp"
#include "core/engine.hpp"
#include "io/input/mongodb_inputformat.hpp"
#include "lib/aggregator_factory.hpp"


int vectorSize = 100; //100 is better
double learningRate = 0.025; //0.025
double starting_learningRate = learningRate;
int numPartitions = 1;
int numIterations = 1;//nodes' parameters are initialized by zero vectors, so in the first batch of the first iteration, the word vectors are not updated.
//int seed = Utils.random.nextLong();
int minCount = 5; //8 is better
int windowSize = 8;
double maxFrequency = 0.005;
int numBatch = 50;

class Term {
  public:
    using KeyT = std::string;
    Term() = default;
    explicit Term(const KeyT& term) : termid(term) {}
    KeyT termid;
    int count;
    const KeyT& id() const {
        return termid;
    }
};

class Document {
   public:
    using KeyT = std::string;
    Document() = default;
    explicit Document(const KeyT& t) : title(t) {}
    KeyT title;
    std::vector<std::string> words;
    int batch;
    //std::string content = "";
    const KeyT& id() const { return title; }

};


class Node {
    public:
        using KeyT = std::string;
        Node() = default;
        explicit Node(const KeyT& node) : nodeid(node) {}
        KeyT nodeid;
        Node* lchild = NULL;
        Node* rchild = NULL;
        Node* parent = NULL;
        int count = 0;
        int location = 0;
        std::vector<double> para;
        const KeyT& id() const {
            return nodeid;
        }
};



inline bool isInteger(const std::string & s){
    if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;
    char * p ;
    strtol(s.c_str(), &p, 10) ;
    return (*p == 0) ;
}
struct comparator {
    bool operator()(Node* i, Node* j) {
        return i->count > j->count;
    }
};

// Get all paras of nodes to update aggregator
void inOrderTraversal(Node* root){
    std::stack<Node*> nodes;
    int i = 0;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            /*
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("test  "+root->id());
            }
            */
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            /*
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("mark  "+std::to_string(root->para));
            }
            */
            root->location = i;
            i++;
            nodes.pop();
            root = root->rchild;
        }
    }
}

// Get the node by id
Node* search(Node* root, std::string nodeid){
    std::stack<Node*> nodes;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            if(root->id() == nodeid) {
                return root;
            }
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            nodes.pop();
            root = root->rchild;
        }
    }
    return NULL;
}

// Get the location of the word in the aggregator vector
/*int find_location(Node* root, std::string nodeid){
    std::stack<Node*> nodes;
    int i = 0;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if(root->id() == nodeid) {
                return i;
            }
            i++;
            nodes.pop();
            root = root->rchild;
        }
    }
    return -1;
}*/

void updateNodesParas(Node* root, std::vector<std::vector<double>> paras){
    // husky::LOG_I << "begin updating";
    std::stack<Node*> nodes;
    int i = 0;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            // husky::LOG_I << "update number: " << paras[i][vectorSize];
            if (paras[i][vectorSize] > 0){
                for (int j = 0; j < vectorSize; j++){
                    root->para[j] += paras[i][j] / paras[i][vectorSize];
                }
            }
            i++;
            nodes.pop();
            root = root->rchild;
        }
    }
}

double logistic_value(std::vector<double> word_node, std::vector<double> context_node){
    double result = 0;
    if (word_node.size() == context_node.size()){
        for (int i = 0; i < word_node.size(); i++){
            result += word_node[i] * context_node[i];
        };
    }
    if(result <= -6){
        return 0;
    }
    else{
        if(result >= 6){
            return 1;
        }
        else{
            return 1.0/(1.0 + exp(0 - result));
        }
    }
}

void train(int count, Node* root, int batch){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    // aggregate the parameters and number of samples
    husky::lib::Aggregator<std::vector<std::vector<double>>> update_paras(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize + 1, 0)),
      [](std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        if(a.size() == b.size()){
            for (int j = 0; j < b.size(); j++) {
                if (a[j].size() == b[j].size()){
                    for (int k = 0; k < a[j].size(); k++){
                        a[j][k] = (a[j][k] + b[j][k]);
                    }
                }
            }
        }
      },
      [&](std::vector<std::vector<double>>& v) {
           v = std::move(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize + 1, 0)));
      });

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    
    list_execute(document_list, {}, {&ac}, [&](Document& doc) {
        if (doc.batch == batch){
            /* learningRate = starting_learningRate * (1 - batch/((float)numBatch+1.0)); 
            if (husky::Context::get_global_tid() == 0) {
                husky::LOG_I << "current learning rate: " << learningRate;
            }*/
            std::vector<Node*> word_nodes_list;
            for (int i = 0; i < doc.words.size(); i++) {
                Node* word_node = search(root, doc.words[i]);
                word_nodes_list.push_back(word_node);
            }
            for (int i = 0; i < doc.words.size(); i++) {
                std::vector<int> context;
                if ( i < windowSize/2) {
                    for (int j = 0; j <= i + windowSize/2; j++){
                        if (j >= doc.words.size()){
                            break;
                        }
                        if ( i != j){
                            context.push_back(j);
                        }
                    }
                }
                else{
                    for (int j = i - windowSize/2; j < i; j++){
                        context.push_back(j);
                    }
                    for (int j = i + 1; j <= i + windowSize/2; j++){
                        if (j >= doc.words.size()){
                            break;
                        }
                        context.push_back(j);
                    }
                }
                /*if (husky::Context::get_global_tid() == 0) {
                    husky::LOG_I << "i:  " << i << "  Mark, after getting context words";
                }*/

                Node* word_node = word_nodes_list[i];
                // Words with little frequency are not in the huffman tree.
                if (word_node == NULL){
                    continue;
                }

                //update nodes on each context path
                std::vector<double> updateWord(vectorSize, 0);
                for (int j = 0; j < context.size(); j++) {
                    Node* context_node = word_nodes_list[context[j]];
                    // Words with little frequency are not in the huffman tree.
                    if (context_node == NULL){
                        continue;
                    }

                    while(context_node->id() != root->id()){
                        double tempValue;
                        /*
                        if (husky::Context::get_global_tid() == 0) {
                            husky::LOG_I << "current context:  " <<context_node->id();
                            husky::LOG_I << "parent rchild id: " << context_node->parent->rchild->id();
                        }*/  

                        if(context_node->id() == context_node->parent->lchild->id()){
                            tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                        }
                        else{
                        
                            tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                        }
                        
                        // husky::LOG_I << "tempValue: " << tempValue;
                        int location = context_node->parent->location;
                        // update node vector
                        update_paras.update_any([&](std::vector<std::vector<double>>& v){
                            for (int m = 0; m < vectorSize; m++) {
                                v[location][m] += tempValue * word_node->para[m];
                            }
                            v[location][vectorSize] += 1;
                        });
                        for (int m = 0; m < vectorSize; m++){
                            updateWord[m] += tempValue*context_node->parent->para[m];
                        }

                        /*
                        if (doc.words[i] == "000"){
                            husky::LOG_I << logistic_value(word_node->para, context_node->parent->para);
                            husky::LOG_I << "temp value" << tempValue;
                        }*/
                        // husky::LOG_I << "finish updating";
                        context_node = context_node->parent;
                        // husky::LOG_I << "context id " << context_node->id();
                        // husky::LOG_I << "parent id " << context_node->parent->id();
                        // husky::LOG_I <<"root id " << root->id();
                    }
                    // husky::LOG_I << "end while";
                }
                int wordLocation = word_node->location;
                update_paras.update_any([&](std::vector<std::vector<double>>& v){
                    for(int m = 0; m < vectorSize; m++){
                        v[wordLocation][m] += updateWord[m];
                    }
                    v[wordLocation][vectorSize] += 1;
                });


            }
            // husky::LOG_I << "finish document " << doc.id();
        
        }//endif
    });
    /*
    if (husky::Context::get_global_tid() == 0) {
        Node* word_node = search(root, "000");
        husky::LOG_I << "before updating word: " << word_node->id() << "  para size: " << word_node->para.size() << " para[0,-1]: " << word_node->para[0] << word_node->para[vectorSize - 1];
    }
*/
    std::vector<std::vector<double>> paras_aggr (update_paras.get_value());
    updateNodesParas(root, paras_aggr);
    /*
    if (husky::Context::get_global_tid() == 0) {
        Node* word_node = search(root, "000");
        int l = find_location(root, "000");
        husky::LOG_I << "paras 000 " << paras_aggr[l][0] << paras_aggr[l][vectorSize - 1];
        husky::LOG_I << "word: " << word_node->id() << "  para size: " << word_node->para.size() << " para[0,-1]: " << word_node->para[0] << word_node->para[vectorSize - 1];
    }*/


    return;
    /*
    if (husky::Context::get_global_tid() == 0) {
        husky::base::log_msg(std::to_string(paras.size()));
        husky::base::log_msg(std::to_string(paras[0][0]));
    }
    */
}

void train_single(int iter, Node* root, int batch){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    // aggregate the parameters and number of samples
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    
    list_execute(document_list, {}, {}, [&](Document& doc) {
        if (doc.batch == batch){
            husky::LOG_I << "batch matches";
            std::vector<Node*> word_nodes_list;
            for (int i = 0; i < doc.words.size(); i++) {
                Node* word_node = search(root, doc.words[i]);
                word_nodes_list.push_back(word_node);
            }
            for (int i = 0; i < doc.words.size(); i++) {
                if (i % 10000 == 0){
                    learningRate = starting_learningRate * (1 - i / (double)(iter * doc.words.size() + 1));
                    if (learningRate < starting_learningRate * 0.0001){
                        learningRate = starting_learningRate * 0.0001;
                    }
                    if (husky::Context::get_global_tid() == 0) {
                        husky::LOG_I << "current learning rate: " << learningRate;
                    }

                }
                std::vector<int> context;
                if ( i < windowSize/2) {
                    for (int j = 0; j <= i + windowSize/2; j++){
                        if (j >= doc.words.size()){
                            break;
                        }
                        if ( i != j){
                            context.push_back(j);
                        }
                    }
                }
                else{
                    for (int j = i - windowSize/2; j < i; j++){
                        context.push_back(j);
                    }
                    for (int j = i + 1; j <= i + windowSize/2; j++){
                        if (j >= doc.words.size()){
                            break;
                        }
                        context.push_back(j);
                    }
                }
                /*if (husky::Context::get_global_tid() == 0) {
                    husky::LOG_I << "i:  " << i << "  Mark, after getting context words";
                }*/

                Node* word_node = word_nodes_list[i];
                // Words with little frequency are not in the huffman tree.
                if (word_node == NULL){
                    continue;
                }

                //update nodes on each context path
                for (int j = 0; j < context.size(); j++) {
                    Node* context_node = word_nodes_list[context[j]];
                    // Words with little frequency are not in the huffman tree.
                    if (context_node == NULL){
                        continue;
                    }

                    std::vector<double> updateWord(vectorSize, 0);
                    while(context_node->id() != root->id()){
                        double tempValue;
                        /*
                        if (husky::Context::get_global_tid() == 0) {
                            husky::LOG_I << "current context:  " <<context_node->id();
                            husky::LOG_I << "parent rchild id: " << context_node->parent->rchild->id();
                        }*/  

                        if(context_node->id() == context_node->parent->lchild->id()){
                            tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                        }
                        else{
                        
                            tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                        }
                         
                        // husky::LOG_I << "tempValue: " << tempValue;
                        int location = context_node->parent->location;
                        // update node vector
                        for (int m = 0; m < vectorSize; m++){
                            updateWord[m] += tempValue*context_node->parent->para[m];
                        }

                        for (int m = 0; m < vectorSize; m++) {
                            context_node->parent->para[m] += tempValue * word_node->para[m];
                        }
                        
                        /*
                        if (doc.words[i] == "000"){
                            husky::LOG_I << logistic_value(word_node->para, context_node->parent->para);
                            husky::LOG_I << "temp value" << tempValue;
                        }*/
                        // husky::LOG_I << "finish updating";
                        context_node = context_node->parent;
                        // husky::LOG_I << "context id " << context_node->id();
                        // husky::LOG_I << "parent id " << context_node->parent->id();
                        // husky::LOG_I <<"root id " << root->id();
                    }
                    for (int m = 0; m < vectorSize; m++){
                        word_node->para[m] += updateWord[m];
                    }
                    // husky::LOG_I << "end while";
                }
                

            }
            // husky::LOG_I << "finish document " << doc.id();
        
        }//endif
    });
    return;
}



std::pair<int, Node*> buildHuffmanTree(auto wc_list){
    /*
    std::vector<std::pair<std::string, int>> wc_list;
    wc_list.push_back(std::make_pair("b", 2));
    wc_list.push_back(std::make_pair("c", 5));
    wc_list.push_back(std::make_pair("a", 2));
    wc_list.push_back(std::make_pair("d", 6));
    */
    std::priority_queue<Node*, std::vector<Node*>, comparator> minHeap;
    srand (time(NULL));
    int id = 0;
    int count = wc_list.size();
    for (int i = 0; i < wc_list.size(); i++){
        if(isInteger(wc_list[i].first)){
            id = std::stoi(wc_list[i].first) + 1;
        }
        Node* node = new Node(wc_list[i].first);
        node->count = wc_list[i].second;
        // para size: vectorSize
        // random initialization
        for (int i = 0; i < vectorSize; i++){
            node->para.push_back((double(rand())/RAND_MAX - 0.5) / vectorSize);
        }
        minHeap.push(node);
    }

    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "after pushing all words";
    }
    while(minHeap.size() > 1) {
        Node* node1 = minHeap.top();
        minHeap.pop();
        Node* node2 = minHeap.top();
        minHeap.pop();
        Node* new_node = new Node(std::to_string(id));
        count++;
        new_node->count = node1->count + node2->count;
        new_node->lchild = node1;
        new_node->rchild = node2;
        for (int i = 0; i < vectorSize; i++){
            new_node->para.push_back(0);
        }
        node1->parent = new_node;
        node2->parent = new_node;
        id++;
        minHeap.push(new_node);
        /*
        if (husky::Context::get_global_tid() == 0) {
            husky::base::log_msg("left:  "+node1->id());
    // buildHuffmanTree(wc_list);
    // buildHuffmanTree(wc_list);
            husky::base::log_msg("right:  "+node2->id());
            husky::base::log_msg("middle:  "+new_node->id());
        }
        */
    }
    Node* top_node = minHeap.top();
    minHeap.pop();
    /*
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "rootid: " + top_node->id() << "  root para:  " << top_node->para[0];
        husky::LOG_I << "rootcount: "+std::to_string(top_node->count);
        husky::LOG_I << "left of root id: "+ top_node->lchild->id() << " left para size: " << top_node->lchild->para.size();
        husky::LOG_I << "root->r->l id: "+top_node->rchild->lchild->id();
    }
    */
    // set location
    inOrderTraversal(top_node);
    std::pair<int, Node*> reValues = std::make_pair(count, top_node);
    return reValues;
    // Testing functions
    if (husky::Context::get_global_tid() == 0) {
        Node* word_node = search(top_node, "hippies");
        husky::LOG_I << "word: " << word_node->id() << "  para size: " << word_node->para.size() << " para[0,-1]: " << word_node->para[0] << word_node->para[vectorSize - 1];
    }
/*
    if (husky::Context::get_global_tid() == 0) {
        int location = find_location(top_node, "hippies");
        husky::LOG_I << "location: " << location;
    }*/
    // std::pair<husky::AggregatorChannel, Node*> updateChannelAndRoot = std::make_pair(ac, top_node);
    // train(ac, top_node);

}



auto createVocab() {
    // term_list: store every term in the corpus
    auto& term_list = husky::ObjListStore::create_objlist<Term>();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "term list id: " << term_list.get_id();
    }
    // document_list: store every document in the corpus for training
    auto& document_list = husky::ObjListStore::create_objlist<Document>();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "document list id: " << document_list.get_id();
    }
    husky::io::MongoDBInputFormat inputformat;
    inputformat.set_server(husky::Context::get_param("mongo_server"));
    inputformat.set_ns(husky::Context::get_param("mongo_db"),husky::Context::get_param("mongo_collection"));
    // inputformat.set_query(mongo::Query("{}").where("this.md5 >= ffa9f46759616718f44cae98d75d77b6"));
    // num_term: channel  for word count
    auto& num_term = husky::ChannelStore::create_push_combined_channel<int, husky::SumCombiner<int>>(inputformat, term_list);
    // wc: aggregator for terms and their counts, pair(term, count)
    // ac: channel for wc
    // Currently, only one aggregator is used here. Since the aggregation is too heavy and it seems to
    // influence the scalability. I have tried to split it into two aggregators, one for terms and one 
    // for counts, but the result will be wrong. Will improve if I find better way to deal with the 
    // problem.
    husky::lib::Aggregator<std::vector<std::pair<std::string, int>>> wc(std::vector<std::pair<std::string, int>>(),
      [](std::vector<std::pair<std::string, int>>& a, const std::vector<std::pair<std::string, int>>& b) {
        for (int i = 0; i < b.size(); i++) {
            a.push_back(b[i]);
        }
    });
   
    auto& ac = husky::lib::AggregatorFactory::get_channel();
    
    // calculate the total number of words
    husky::lib::Aggregator<int> num_total_words(0,
        [](int& a, const int& b){ a += b; },
        0
    );

    auto& num_channel = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::ChannelBase*> out_channel;
    out_channel.push_back(&num_channel);
    out_channel.push_back(&num_term); 
    /*
    husky::lib::Aggregator<std::vector<int>> wc(std::vector<int>(),
      [](std::vector<int>& a, const std::vector<int>& b) {
        for (int i = 0; i < b.size(); i++) {
            a.push_back(b[i]);
        }
    });

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    
    husky::lib::Aggregator<std::vector<std::string>> term_aggregator (std::vector<std::string>(),
      [](std::vector<std::string>& a, const std::vector<std::string>& b) {
        for (int i = 0; i < b.size(); i++) {
            a.push_back(b[i]);
        }
    });

    auto& term_channel = husky::lib::AggregatorFactory::get_channel();*/

    auto parse = [&] (std::string& chunk) {
        mongo::BSONObj o = mongo::fromjson(chunk);
        Document doc(o.getStringField(husky::Context::get_param("doc_id")));
        std::string content = o.getStringField(husky::Context::get_param("doc_content"));
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t\n.,()\'\":;!?<>[]-");
        boost::tokenizer<boost::char_separator<char>> tok(content, sep);
        for (auto& w : tok) {
            doc.words.push_back(w);
            std::transform(doc.words.back().begin(), doc.words.back().end(), doc.words.back().begin(), ::tolower); 
            //doc.content += " "+w;
            num_term.push(1, doc.words.back());
            num_total_words.update(1);
        }
        doc.batch = rand() % numBatch;
        document_list.add_object(doc);
    }; 
    load(inputformat, out_channel, parse);
    int maxCount = (int) num_total_words.get_value() * maxFrequency;
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "maxCount: " << maxCount;
    }

    // husky::LOG_I << " Finish loading ";
    list_execute(term_list, {&num_term}, {&ac}, [&](Term& t){
        t.count = num_term.get(t);
        if(t.count > maxCount){
            husky::LOG_I << "word " << t.id() << " ,count " << t.count;
        }
        if(t.count > minCount && t.count < maxCount){
            wc.update_any([&](std::vector<std::pair<std::string, int>>& v){
                v.push_back(std::make_pair(t.id(), t.count));
            });
        }
        /*
        if (t.count > minCount) {
            wc.update_any([&](std::vector<int>& v){
                v.push_back(t.count);
            });
            term_aggregator.update_any([&](std::vector<std::string>& v){
                v.push_back(t.id());
            });
        }*/

    });
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << " Testing1 ";
    }
    // Drop term list
    husky::ObjListStore::drop_objlist(0);
 
    // husky::LOG_I << " After dropping list ";

    // Here, the word count has finished and wc_list is a list of (term, count).
    auto wc_list (wc.get_value());
 
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "wc_list size: "<< wc_list.size();
    }
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << wc_list.at(0).first;
        //  husky::LOG_I << wc_list.at(1).first;
        for (int i = 0; i < wc_list.size(); i++) {
            if (wc_list.at(i).first == "OpenCV"){
                husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
            }
            else{
                if (wc_list.at(i).first == "hippies"){
                    husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
                }
            }
        }
    }

/*
    auto ta_list (term_aggregator.get_value());

    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < wc_list.size(); i++){
            if (ta_list.at(i)=="OpenCV"){
                husky::LOG_I << "Word: " << ta_list.at(i) << "  Count: " << wc_list.at(i);
            }
            else{
                if (ta_list.at(i)=="hippies"){
                    husky::LOG_I << "Word: " << ta_list.at(i) << "  Count: " << wc_list.at(i);
                }
            }
        }
    }
*/
/*    
    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < wc_list.size(); i = i + wc_list.size()/8) {
            husky::LOG_I << "Count:  " + std::to_string(wc_list.at(i));
        }
    }
    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < ta_list.size(); i = i + ta_list.size()/8) {
            husky::LOG_I << "Word:  " + ta_list.at(i);
        }
    }
    if (husky::Context::get_global_tid() == 0){
        husky::LOG_I << " Testing2 ";
    }
*/
/*
    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < wc_list.size(); i = i + wc_list.size()/800) {
            husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
        }
    }
*/
    // wc_list: list of (term, count) in the order of count
    std::sort(wc_list.begin(), wc_list.end(), [](std::pair<std::string, int> a, std::pair<std::string, int> b) {
            return a.second > b.second;});

    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < wc_list.size(); i++) {
            if (wc_list.at(i).first == "OpenCV"){
                husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
            }
            else{
                if (wc_list.at(i).first == "hippies"){
                    husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
             
                }
            }
        }
    }
    
    return wc_list;
    // buildHuffmanTree(wc_list);
}

auto createVocabByFile(char* filename) {
    // term_list: store every term in the corpus
    auto& term_list = husky::ObjListStore::create_objlist<Term>();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "term list id: " << term_list.get_id();
    }
    // document_list: store every document in the corpus for training
    auto& document_list = husky::ObjListStore::create_objlist<Document>();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "document list id: " << document_list.get_id();
    }
    // num_term: channel  for word count
    auto& num_term = husky::ChannelStore::create_push_combined_channel<int, husky::SumCombiner<int>>(document_list, term_list);
    // wc: aggregator for terms and their counts, pair(term, count)
    // ac: channel for wc
    husky::lib::Aggregator<std::vector<std::pair<std::string, int>>> wc(std::vector<std::pair<std::string, int>>(),
      [](std::vector<std::pair<std::string, int>>& a, const std::vector<std::pair<std::string, int>>& b) {
        for (int i = 0; i < b.size(); i++) {
            a.push_back(b[i]);
        }
    });
    auto& ac = husky::lib::AggregatorFactory::get_channel();
    
    // calculate the total number of words
    int num_total_words = 0; 
    FILE * file;
    long fileSize;
    char * content;
    size_t result;
    Document doc("doc1");
    file = fopen (filename , "r");

    // obtain file size:
    fseek (file , 0 , SEEK_END);
    fileSize = ftell (file);
    husky::LOG_I << "file size: " << fileSize;
    rewind (file);
    // allocate memory to contain the whole file:
    content = (char*) malloc (sizeof(char)*fileSize);
    // copy the file into the buffer:
    result = fread (content,1,fileSize,file);

    husky::LOG_I << "after reading file";
    std::string word;
    std::istringstream s(content);
    
    free(content);
    fclose(file);
    husky::LOG_I << "before loop";
    while(s >> word){
        std::transform(word.begin(), word.end(), word.begin(), ::tolower); 
        num_total_words += 1;
        doc.words.push_back(word);
    }
    doc.batch = 0;
    document_list.add_object(doc);

    husky::LOG_I << "after adding document";
    list_execute(document_list, {}, {&num_term}, [&](Document& doc){
        for (std::string w : doc.words){
            num_term.push(1, w);
        }
    });

    husky::LOG_I << "after pushing";
    int maxCount = (int) num_total_words * maxFrequency;
    list_execute(term_list, {&num_term}, {&ac}, [&](Term& t){
        t.count = num_term.get(t);
        if(t.count > maxCount){
            husky::LOG_I << "word " << t.id() << " ,count " << t.count;
        }
        if(t.count > minCount && t.count < maxCount){
            wc.update_any([&](std::vector<std::pair<std::string, int>>& v){
                v.push_back(std::make_pair(t.id(), t.count));
            });
        }
    });
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << " Testing1 ";
    }
    // Drop term list
    husky::ObjListStore::drop_objlist(0);
    
    auto wc_list (wc.get_value());
 
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "wc_list size: "<< wc_list.size();
    }
    
    std::sort(wc_list.begin(), wc_list.end(), [](std::pair<std::string, int> a, std::pair<std::string, int> b) {
            return a.second > b.second;});

    if (husky::Context::get_global_tid() == 0) {
        for (int i = 0; i < wc_list.size(); i++) {
            if (wc_list.at(i).first == "OpenCV"){
                husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
            }
            else{
                if (wc_list.at(i).first == "hippies"){
                    husky::LOG_I << "Word: " + wc_list.at(i).first << "    Count: "<<wc_list.at(i).second;
             
                }
            }
        }
    }
    
    return wc_list;


    // buildHuffmanTree(wc_list);
}



std::string test(std::string w1, std::string w2, std::string w3, Node* root){
    Node* node1 = search(root, w1); 
    Node* node2 = search(root, w2); 
    Node* node3 = search(root, w3);
    std::string result;
    double v1 = 0;
    double v2 = 0;
    double v3 = 0;
    double sim = 0;
    double maxSim = 0;
    std::stack<Node*> nodes;
    std::vector<double> para;
    if (husky::Context::get_global_tid() == 0){
        root = root->lchild->lchild;
    }
    if (husky::Context::get_global_tid() == 1){
        root = root->lchild->rchild;
    }
    if (husky::Context::get_global_tid() == 2){
        root = root->rchild->lchild;
    }
    if (husky::Context::get_global_tid() == 3){
        root = root->rchild->lchild;
    }

    for (int i = 0; i < vectorSize; i++){
        para.push_back(node1->para[i] - node2->para[i] + node3->para[i]);
        v3 += (node1->para[i] - node2->para[i] + node3->para[i])*(node1->para[i] - node2->para[i] + node3->para[i]);
    }
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if (root->lchild == NULL && root->rchild == NULL &&root->id() != w1 && root->id() != w2 && root->id()!= w3){
                for(int i = 0; i < vectorSize; i++){
                    v1 += para[i] * root->para[i];
                    v2 += root->para[i] * root->para[i];
                }

                sim = v1/sqrt(v2);
                if (sim > maxSim){
                    maxSim = sim;
                    result = root->id();
                    husky::LOG_I << "word: " << result << "  sim: " << sim/sqrt(v3);
                }
                v1 = 0;
                v2 = 0;
            }
            nodes.pop();
            root = root->rchild;
 
        }
    }
    return result;

}

std::string test_v2(std::string w1, std::string w2, std::string w3, Node* root){
    Node* node1 = search(root, w1); 
    Node* node2 = search(root, w2); 
    Node* node3 = search(root, w3);
    std::string result;
    double v1 = 0;
    double v2 = 0;
    double v3 = 0;
    double sim = 0;
    double maxSim = 0;
    std::stack<Node*> nodes;
    std::vector<double> para;
    if (husky::Context::get_global_tid() == 20){
        root = root->lchild->lchild;
    }
    if (husky::Context::get_global_tid() == 21){
        root = root->lchild->rchild;
    }
    if (husky::Context::get_global_tid() == 22){
        root = root->rchild->lchild;
    }
    if (husky::Context::get_global_tid() == 23){
        root = root->rchild->lchild;
    }

    for (int i = 0; i < vectorSize; i++){
        para.push_back(node1->para[i] - node2->para[i]);
        v3 += (node1->para[i] - node2->para[i])*(node1->para[i] - node2->para[i]);
    }
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if (root->lchild == NULL && root->rchild == NULL &&root->id() != w1 && root->id() != w2 && root->id()!= w3){
                for(int i = 0; i < vectorSize; i++){
                    v1 += para[i] * (root->para[i] - node3->para[i]);
                    v2 += (root->para[i] - node3->para[i]) * (root->para[i] - node3->para[i]);
                }

                sim = v1/sqrt(v2);
                if (sim > maxSim){
                    maxSim = sim;
                    result = root->id();
                    husky::LOG_I << "word v2 : " << result << "  sim: " << sim/sqrt(v3);
                }
                v1 = 0;
                v2 = 0;
            }
            nodes.pop();
            root = root->rchild;
 
        }
    }
    return result;

}


std::string test_normalized(std::string w1, std::string w2, std::string w3, Node* root){
    Node* node1 = search(root, w1); 
    Node* node2 = search(root, w2); 
    Node* node3 = search(root, w3);
    std::string result;
    double v1 = 0;
    double v2 = 0;
    double v3 = 0;
    double sim = 0;
    double maxSim = 0;
    std::stack<Node*> nodes;
    std::vector<double> para;
    if (husky::Context::get_global_tid() == 4){
        root = root->lchild->lchild;
    }
    if (husky::Context::get_global_tid() == 5){
        root = root->lchild->rchild;
    }
    if (husky::Context::get_global_tid() == 6){
        root = root->rchild->lchild;
    }
    if (husky::Context::get_global_tid() == 7){
        root = root->rchild->lchild;
    }

    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    std::vector<double> para1(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para1[i] = node1->para[i];
        sum1 += para1[i];
    }
    std::vector<double> para2(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para2[i] = node2->para[i];
        sum2 += para2[i];
    }
    std::vector<double> para3(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para3[i] = node3->para[i];
        sum3 += para3[i];

    }
        
    for (int i = 0; i < vectorSize; i++){
        para1[i] = para1[i]/sum1;
        para2[i] = para2[i]/sum2;
        para3[i] = para3[i]/sum3;
    }

    for (int i = 0; i < vectorSize; i++){
        para.push_back(para1[i] - para2[i] + para3[i]);
        v3 += (para1[i] - para2[i] + para3[i])*(para1[i] - para2[i] + para3[i]);
    }
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if (root->lchild == NULL && root->rchild == NULL &&root->id() != w1 && root->id() != w2 && root->id()!= w3){
                double sum4 = 0;
                std::vector<double> para4(vectorSize, 0);
                for (int i = 0; i < vectorSize; i++){
                    para4[i] = root->para[i];
                    sum4 += para4[i];
                }
                for (int i = 0; i < vectorSize; i++){
                    para4[i] = para4[i]/sum4;
                }

                for(int i = 0; i < vectorSize; i++){
                    v1 += para[i] * para4[i];
                    v2 += para4[i] * para4[i];
                }

                sim = v1/sqrt(v2);
                if (sim > maxSim){
                    maxSim = sim;
                    result = root->id();
                    husky::LOG_I << "normalized word: " << result << "  sim: " << sim/sqrt(v3);
                }
                v1 = 0;
                v2 = 0;
            }
            nodes.pop();
            root = root->rchild;
 
        }
    }
    return result;

}
std::string test_normalized_v2(std::string w1, std::string w2, std::string w3, Node* root){
    Node* node1 = search(root, w1); 
    Node* node2 = search(root, w2); 
    Node* node3 = search(root, w3);
    std::string result;
    double v1 = 0;
    double v2 = 0;
    double v3 = 0;
    double sim = 0;
    double maxSim = 0;
    std::stack<Node*> nodes;
    std::vector<double> para;
    if (husky::Context::get_global_tid() == 24){
        root = root->lchild->lchild;
    }
    if (husky::Context::get_global_tid() == 25){
        root = root->lchild->rchild;
    }
    if (husky::Context::get_global_tid() == 26){
        root = root->rchild->lchild;
    }
    if (husky::Context::get_global_tid() == 27){
        root = root->rchild->lchild;
    }

    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    std::vector<double> para1(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para1[i] = node1->para[i];
        sum1 += para1[i];
    }
    std::vector<double> para2(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para2[i] = node2->para[i];
        sum2 += para2[i];
    }
    std::vector<double> para3(vectorSize, 0);
    for (int i = 0; i < vectorSize; i++){
        para3[i] = node3->para[i];
        sum3 += para3[i];

    }
        
    for (int i = 0; i < vectorSize; i++){
        para1[i] = para1[i]/sum1;
        para2[i] = para2[i]/sum2;
        para3[i] = para3[i]/sum3;
    }

    for (int i = 0; i < vectorSize; i++){
        para.push_back(para1[i] - para2[i]);
        v3 += (para1[i] - para2[i])*(para1[i] - para2[i]);
    }
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if (root->lchild == NULL && root->rchild == NULL &&root->id() != w1 && root->id() != w2 && root->id()!= w3){
                double sum4 = 0;
                std::vector<double> para4(vectorSize, 0);
                for (int i = 0; i < vectorSize; i++){
                    para4[i] = root->para[i];
                    sum4 += para4[i];
                }
                for (int i = 0; i < vectorSize; i++){
                    para4[i] = para4[i]/sum4;
                }

                for(int i = 0; i < vectorSize; i++){
                    v1 += para[i] * (para4[i] - para3[i]);
                    v2 += (para4[i] - para3[i]) * (para4[i] - para3[i]);
                }

                sim = v1/sqrt(v2);
                if (sim > maxSim){
                    maxSim = sim;
                    result = root->id();
                    husky::LOG_I << "normalized word v2: " << result << "  sim: " << sim/sqrt(v3);
                }
                v1 = 0;
                v2 = 0;
            }
            nodes.pop();
            root = root->rchild;
 
        }
    }
    return result;

}


void distance(Node* root, std::string nodeid, int topk){
    double v1 = 0;
    double v2 = 0;
    double v3 = 0;
    double sim;
    std::priority_queue<std::pair<std::string,double>, std::vector<std::pair<std::string, double>>,
        std::function<bool(std::pair<std::string, double>,std::pair<std::string, double>)>> result (
        [](std::pair<std::string, double> a, std::pair<std::string, double> b) {
            return a.second < b.second;});
    Node* word_node = search(root, nodeid);
    for (int i = 0; i < vectorSize; i++){
        v3 += word_node->para[i] * word_node->para[i];
    }
    std::stack<Node*> nodes;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            nodes.pop();
            if (root->lchild == NULL && root->rchild == NULL){
                for(int i = 0; i < vectorSize; i++){
                    v1 += word_node->para[i] * root->para[i];
                    v2 += root->para[i] * root->para[i];
                }
                sim = v1/(sqrt(v2)*sqrt(v3));
                v1 = 0;
                v2 = 0;
                result.push(std::make_pair(root->id(), sim));
            }
            root = root->rchild;
        }
    }
    for(int i = 0; i < topk; i++){
        husky::LOG_I << nodeid <<" closest word" << i <<": " << result.top().first << "  sim: " << result.top().second;
        result.pop();
        husky::LOG_I << nodeid << "[" << i << "] " << word_node->para[i];
    }
    std::pair<std::string,double> far;
    while(!result.empty()){
        far = result.top();
        result.pop();
    }
        husky::LOG_I << nodeid <<" farthest word: " << far.first << "  sim: " << far.second;

}
void word2vec() {
    if (husky::Context::get_global_tid() == 0) {
        ProfilerStart("/tmp/jiayi/a.prof");
    }
    //single machine
    /* 
    auto wc_list = createVocabByFile("/data/jiayi/husky_github/husky/word2vec/trunk/text8");
    husky::LOG_I << "wc_list size: " << wc_list.size();
    if (husky::Context::get_local_tid() == 0){
        // husky::LOG_I << "test local id";
    }
    auto rePair = buildHuffmanTree(wc_list);
    int numNodes = rePair.first;
    Node* root = rePair.second;
    husky::LOG_I << "num of nodes: " << numNodes;


    for (int k = 0; k < numIterations; k++){
        distance(root, "paris", 10);
        distance(root, "windows", 10);
        train_single(k + 1, root, 0);
        //test word vectors
        if (husky::Context::get_global_tid() == 0){
            distance(root, "paris", 10);
            distance(root, "windows", 10);
        }
        if (husky::Context::get_global_tid() == 0) {
            husky::LOG_I << "after iteration" << k;
        }

    }
    return;
    */
    //multiple machines
   
    auto wc_list = createVocab();
    auto rePair = buildHuffmanTree(wc_list);
    int numNodes = rePair.first;
    Node* root = rePair.second;
    if(husky::Context::get_global_tid() == 0){
        husky::LOG_I << "test";
   } 
    for (int k = 0; k < numIterations; k++){
        // before training
        
        if (husky::Context::get_global_tid() == 0) {
            Node* node1 = search(root, "man"); 
            Node* node2 = search(root, "woman"); 
            Node* node3 = search(root, "girl"); 
            Node* node4 = search(root, "boy");
            double v1 = 0;
            double v2 = 0;
            double v3 = 0;
            double sim = 0;
            for(int i = 0; i < vectorSize; i++){
                // husky::LOG_I << "man - woman " << i << " : " <<node1->para[i] - node2->para[i];
                // husky::LOG_I << "boy - girl " << i << " : " <<node4->para[i] - node3->para[i];
                v1 += (node1->para[i] - node2->para[i]) * (node4->para[i] - node3->para[i]);
                v2 += (node1->para[i] - node2->para[i]) * (node1->para[i] - node2->para[i]);
                v3 += (node4->para[i] - node3->para[i]) * (node4->para[i] - node3->para[i]);    
            }
            sim = v1/(sqrt(v2) * sqrt(v3));
            husky::LOG_I << "before training, similarity1:  " << sim;
            if (husky::Context::get_global_tid() == 0){
                distance(root, "paris", 10);
                distance(root, "windows", 10);
            }

        }
        
        for(int j = 0; j < numBatch; j++){
            train(numNodes, root, j);
            if (husky::Context::get_global_tid() == 0) {
                husky::LOG_I << "after training batch" << j;
            }
            //test word vectors

            if (husky::Context::get_global_tid() == 0){
                distance(root, "paris", 10);
                distance(root, "windows", 10);
            }
            if (husky::Context::get_global_tid() == 0) {

                Node* node1 = search(root, "man");
                // husky::LOG_I << "after searching man";
                // husky::LOG_I << "node1 id" << node1->id();

                Node* node2 = search(root, "woman"); 
                // husky::LOG_I << "node2 id" << node2->id();
                Node* node3 = search(root, "girl");
                // husky::LOG_I << "node3 id" << node3->id();
                Node* node4 = search(root, "boy");
                // husky::LOG_I << "after searching nodes.";
                double v1 = 0;
                double v2 = 0;
                double v3 = 0;
                double sim = 0;
                for(int i = 0; i < vectorSize; i++){
                    // husky::LOG_I << "man - woman " << i << " : " <<node1->para[i] - node2->para[i];
                    // husky::LOG_I << "boy - girl " << i << " : " <<node4->para[i] - node3->para[i];
                    v1 +=(node1->para[i] - node2->para[i]) * (node4->para[i] - node3->para[i]);
                    v2 += (node1->para[i] - node2->para[i]) * (node1->para[i] - node2->para[i]);
                    v3 += (node4->para[i] - node3->para[i]) * (node4->para[i] - node3->para[i]);    
                }
                sim = v1/(sqrt(v2) * sqrt(v3));
                husky::LOG_I << "after training " << j <<", similarity:  " << sim;

            }
        }

    }
    return;
    std::string w1 = "man";
    std::string w2 = "woman";
    std::string w3 = "girl";
    if (husky::Context::get_global_tid() <= 3){
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result1: " << result;
    }
    
    if (husky::Context::get_global_tid() > 3 && husky::Context::get_global_tid() <= 7){
        std::string result = test_normalized(w1, w2, w3, root);
        husky::LOG_I << "Result1_normalized: " << result;
    }
    if (husky::Context::get_global_tid() >= 20 && husky::Context::get_global_tid() <= 23){
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result1_v2: " << result;
    }
    
    if (husky::Context::get_global_tid() > 23 && husky::Context::get_global_tid() <= 27){
        std::string result = test_normalized_v2(w1, w2, w3, root);
        husky::LOG_I << "Result1_normalized_v2: " << result;
    }
    
    w1 = "number";
    w2 = "numbers";
    w3 = "vectors";
    if (husky::Context::get_global_tid() <= 3){
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result2: " << result;
    }
    
    if (husky::Context::get_global_tid() > 3 && husky::Context::get_global_tid() <= 7){
        std::string result = test_normalized(w1, w2, w3, root);
        husky::LOG_I << "Result2_normalized: " << result;
    }
    if (husky::Context::get_global_tid() >= 20 && husky::Context::get_global_tid() <= 23){
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result2_v2: " << result;
    }
    
    if (husky::Context::get_global_tid() > 23 && husky::Context::get_global_tid() <= 27){
        std::string result = test_normalized_v2(w1, w2, w3, root);
        husky::LOG_I << "Result2_normalized_v2: " << result;
    }


    
    /*
    if (husky::Context::get_global_tid() == 0){
        std::string result = test("man", "woman", "girl", root->lchild->lchild);
        husky::LOG_I << "result: " << result;
    }
    if (husky::Context::get_global_tid() == 1){
        std::string result = test("man", "woman", "girl", root->lchild->rchild);
        husky::LOG_I << "result: " << result;
    }
    if (husky::Context::get_global_tid() == 2){
        std::string result = test("man", "woman", "girl", root->rchild->lchild);
        husky::LOG_I << "result: " << result;
    }
    if (husky::Context::get_global_tid() == 3){
        std::string result = test("man", "woman", "girl", root->rchild->rchild);
        husky::LOG_I << "result: " << result;
    }
*/
    if (husky::Context::get_global_tid() == 0){
        ProfilerStop();
    }

  
}




int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("mongo_server");
    args.push_back("mongo_db");
    args.push_back("mongo_collection");
    args.push_back("mongo_user");
    args.push_back("mongo_pwd");
    args.push_back("doc_id");
    args.push_back("doc_content");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(word2vec);
        return 0;
    }
    return 1;
}
