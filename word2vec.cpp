#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <functional>
#include <cmath>
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
double learningRate = 0.025;
int numPartitions = 1;
int numIterations = 3;//nodes' parameters are initialized by zero vectors, so in the first iteration, the word vectors are not updated.
//int seed = Utils.random.nextLong();
int minCount = 5; //8 is better
int windowSize = 5;

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
    std::stack<Node*> nodes;
    int i = 0;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            for (int j = 0; j < vectorSize; j++){
                root->para[j] += paras[i][j];
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
            return 1.0/(1.0 + exp(result));
        }
    }
}

void train(int count, Node* root){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    husky::lib::Aggregator<std::vector<std::vector<double>>> update_paras(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)),
      [](std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        if(a.size() == b.size()){
            for (int j = 0; j < b.size(); j++) {
                if (a[j].size() == b[j].size()){
                    for (int k = 0; k < a[j].size(); k++){
                        a[j][k] = (a[j][k] + b[j][k])/2;
                    }
                }
            }
        }
      },
      [&](std::vector<std::vector<double>>& v) {
           v = std::move(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)));
      });

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    list_execute(document_list, {}, {&ac}, [&](Document& doc) {
        for (int i = 0; i < doc.words.size(); i++) {
            std::vector<std::string> context;
            if ( i < windowSize/2) {
                for (int j = 0; j <= i + windowSize/2; j++){
                    if (j >= doc.words.size()){
                        break;
                    }
                    if ( i != j){
                        context.push_back(doc.words[j]);
                    }
                }
            }
            else{
                for (int j = i - windowSize/2; j < i; j++){
                    context.push_back(doc.words[j]);
                }
                for (int j = i + 1; j <= i + windowSize/2; j++){
                    if (j >= doc.words.size()){
                        break;
                    }
                    context.push_back(doc.words[j]);
                }
            }
            /*if (husky::Context::get_global_tid() == 0) {
                husky::LOG_I << "i:  " << i << "  Mark, after getting context words";
            }*/

            Node* word_node = search(root, doc.words[i]);
            // Words with little frequency are not in the huffman tree.
            if (word_node == NULL){
                continue;
            }

            //update nodes on each context path
            std::vector<double> updateWord(vectorSize, 0);
            for (int j = 0; j < context.size(); j++) {
                Node* context_node = search(root, context[j]);
                // Words with little frequency are not in the huffman tree.
                if (context_node == NULL){
                    continue;
                }

                while(context_node->parent->id() != root->id()){
                    double tempValue;
                    /*
                    if (husky::Context::get_global_tid() == 0) {
                        husky::LOG_I << "current context:  " <<context_node->id();
                        husky::LOG_I << "parent rchild id: " << context_node->parent->rchild->id();
                    }*/  

                    if(context_node == context_node->parent->lchild){
                        tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                    }
                    else{
                    
                        tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                    }
                    
                    // husky::LOG_I << "tempValue: " << tempValue;
                    int location = context_node->parent->location;
                    // update node vector
                    for(int m = 0; m < vectorSize; m++){
                        update_paras.update_any([&](std::vector<std::vector<double>>& v){
                                v[location][m] += tempValue * word_node->para[m];
                        });
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
            for(int m = 0; m < vectorSize; m++){
                update_paras.update_any([&](std::vector<std::vector<double>>& v){
                    v[wordLocation][m] += updateWord[m];
                });
            }


        }
        // husky::LOG_I << "finish document " << doc.id();
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
        boost::char_separator<char> sep(" \t\n.,()\'\":;!?<>[]");
        boost::tokenizer<boost::char_separator<char>> tok(content, sep);
        for (auto& w : tok) {
            doc.words.push_back(w);
            //doc.content += " "+w;
            num_term.push(1, w);
        }
        document_list.add_object(doc);
    }; 
    load(inputformat, parse);

    // husky::LOG_I << " Finish loading ";
    list_execute(term_list, {&num_term}, {&ac}, [&](Term& t){
        t.count = num_term.get(t);
        if(t.count > minCount){
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

void word2vec() {
    if (husky::Context::get_global_tid() == 0) {
        ProfilerStart("/tmp/jiayi/a.prof");
    }
    auto wc_list = createVocab();
    auto rePair = buildHuffmanTree(wc_list);
    int numNodes = rePair.first;
    Node* root = rePair.second;
    
    
    for (int i = 0; i < numIterations; i++){
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
                v1 +=(node1->para[i] - node2->para[i]) * (node4->para[i] - node3->para[i]);
                v2 += (node1->para[i] - node2->para[i]) * (node1->para[i] - node2->para[i]);
                v3 += (node4->para[i] - node3->para[i]) * (node4->para[i] - node3->para[i]);    
            }
            sim = v1/(sqrt(v2) * sqrt(v3));
            husky::LOG_I << "before training, similarity1:  " << sim;
        }
        
        train(numNodes, root);
        //test word vectors
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
                v1 +=(node1->para[i] - node2->para[i]) * (node4->para[i] - node3->para[i]);
                v2 += (node1->para[i] - node2->para[i]) * (node1->para[i] - node2->para[i]);
                v3 += (node4->para[i] - node3->para[i]) * (node4->para[i] - node3->para[i]);    
            }
            sim = v1/(sqrt(v2) * sqrt(v3));
            husky::LOG_I << "after training, similarity2:  " << sim;

        }

    }
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
