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


int numWorkers = 320 ;
int vectorSize = 100; //100 is better
double learningRate = 0.025; //0.025 for single thread; 0.025*10 for 300 threads
double starting_learningRate = learningRate;
int numPartitions = 100000;
int numIterations = 30;//nodes' parameters are initialized by zero vectors, so in the first batch of the first iteration, the word vectors are not updated.
//int seed = Utils.random.nextLong();
int minCount = 8; //8 is better
int windowSize = 8;
double maxFrequency = 0.005; // 0.005
int numBatch = 1;

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

    friend husky::BinStream& operator<<(husky::BinStream& stream, const Document& doc) {
        stream << doc.title << doc.words << doc.batch;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Document& doc) {
        stream >> doc.title >> doc.words >> doc.batch;
        return stream;
    }

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

auto createMap(Node* root){
    std::unordered_map<std::string, Node*> map;
    std::stack<Node*> nodes;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            if(root->lchild == NULL && root->rchild == NULL){
                std::pair<std::string,Node*> ele (root->id(),root);
                map.insert (ele);  
            }
            nodes.pop();
            root = root->rchild;
        }
    }
    return map;
}

// Get all paras of nodes to update aggregator
void inOrderTraversal(Node* root){
    std::stack<Node*> nodes;
    int i = 0;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
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
/*
int find_location(Node* root, std::string nodeid){
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
                    root->para[j] = paras[i][j]/(double)numWorkers;
                }
            i++;
            nodes.pop();
            root = root->rchild;
        }
    }
    
}

void updateNodesParasSingle(Node* root, std::vector<std::vector<double>> paras){
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
                root->para[j] = paras[i][j];
            }

            i++;
            nodes.pop();
            root = root->rchild;
        }
    }
    
}


std::vector<std::vector<double>> getNodesParas(Node* root){
    std::vector<std::vector<double>> result;
    std::stack<Node*> nodes;
    while (root != NULL || !nodes.empty()){
        if(root != NULL) {
            nodes.push(root);
            root = root->lchild;
        }
        else{
            root = nodes.top();
            result.push_back(root->para);
            nodes.pop();
            root = root->rchild;
        }
    }
    return result;
    
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
void train_v2(std::unordered_map<std::string, Node*> map, int count, Node* root){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    // aggregate the parameters and number of samples
    husky::lib::Aggregator<std::vector<std::vector<double>>> update_paras(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)),
      [](std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        if(a.size() == b.size()){
            for (int j = 0; j < b.size(); j++) {
                if (a.at(j).size() == b.at(j).size()){
                    for (int k = 0; k < a.at(j).size(); k++){
                        a.at(j).at(k) = a.at(j).at(k) + b.at(j).at(k);
                    }
                }
            }
        }
      },
      [&](std::vector<std::vector<double>>& v) {
           v = std::move(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)));
      });

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    update_paras.to_reset_each_iter(); 
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    for (int m = 0; m < numBatch; m++){
        list_execute(document_list, {}, {&ac}, [&](Document& doc) {
            if (doc.batch == m){
                std::vector<Node*> word_nodes_list;
                for (int i = 0; i < doc.words.size(); i++) {
                    std::unordered_map<std::string,Node*>::const_iterator got = map.find (doc.words.at(i));
                    if ( got == map.end() ){
                        word_nodes_list.push_back(NULL);
                        }
                    else{
                        word_nodes_list.push_back(got->second);
                    }
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
                    

                    Node* word_node = word_nodes_list.at(i);
                    // Words with little frequency are not in the huffman tree.
                    if (word_node == NULL){
                        continue;
                    }

                    //update nodes on each context path
                    //a < window * 2 + 1 - b
                    for (int j = 0; j < context.size() ; j++) {
                        Node* context_node = word_nodes_list.at(context.at(j));
                        
                        // Words with little frequency are not in the huffman tree.
                        if (context_node == NULL){
                            continue;
                        }

                        std::vector<double> updateWord(vectorSize, 0);
                        while(context_node != root){
                            double tempValue;

                            if(context_node->id() == context_node->parent->lchild->id()){
                                tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                            }
                            else{
                            
                                tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                            }
                            
                            Node* parent = context_node->parent; 
                            int location = parent->location;
                            // update node vector
                            for (int a = 0; a < vectorSize; a++){
                                updateWord.at(a) += tempValue*parent->para.at(a);
                            }

                            for (int a = 0; a < vectorSize; a++) {
                                parent->para.at(a) += tempValue * word_node->para.at(a);
                            }
                            context_node = parent;
                        }
                        for(int a = 0; a < vectorSize; a++){
                            word_node->para.at(a) += updateWord.at(a);
                        }
                    }
                }
            
            }//endif
        });
        if(husky::Context::get_global_tid() == 0 ){
             husky::LOG_I << "after list execute";
        }

        std::vector<std::vector<double>> paras = getNodesParas(root);
        if(husky::Context::get_global_tid() == 0 ){
             husky::LOG_I << "after getting paras";
        }

        update_paras.update_any([&](std::vector<std::vector<double>>& v){
            for(int k = 0; k < count; k++){
                for(int a = 0; a < vectorSize; a++){
                    v.at(k).at(a) = paras.at(k).at(a);
                }
            }
        });
        if(husky::Context::get_global_tid() == 0 ){
             husky::LOG_I << "after updating any";
        }

        
        husky::lib::AggregatorFactory::sync();
        if(husky::Context::get_global_tid() == 0 ){
             husky::LOG_I << "after sync";
        }
        std::vector<std::vector<double>> paras_aggr (update_paras.get_value());
        updateNodesParas(root, paras_aggr);
         
        if (husky::Context::get_global_tid() == 0){
            distance(root, "graduate", 10);
            distance(root, "windows", 10);
            distance(root, "english", 10);

            husky::LOG_I << "end batch " << m; 
        }
        

    }


    return;
}


void train(int count, Node* root){
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
    update_paras.to_reset_each_iter(); 
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    for (int m = 0; m < numBatch; m++){ 
        list_execute(document_list, {}, {&ac}, [&](Document& doc) {
            if (doc.batch == m){
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

                            if(context_node->id() == context_node->parent->lchild->id()){
                                tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                            }
                            else{
                            
                                tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                            }
                            
                            int location = context_node->parent->location;
                            // update node vector
                            
                            for (int m = 0; m < vectorSize; m++){
                                updateWord[m] += tempValue*context_node->parent->para[m];
                            }

                            update_paras.update_any([&](std::vector<std::vector<double>>& v){
                                for (int m = 0; m < vectorSize; m++) {
                                    v[location][m] += tempValue * word_node->para[m];
                                }
                                v[location][vectorSize] += 1;
                            });
                            context_node = context_node->parent;
                        }
                        int wordLocation = word_node->location;
                        update_paras.update_any([&](std::vector<std::vector<double>>& v){
                            for(int m = 0; m < vectorSize; m++){
                                v[wordLocation][m] += updateWord[m];
                            }
                            v[wordLocation][vectorSize] += 1;
                        });
                    }

                    
                }
            
            }//endif

        });
        
        Node* word_node = search(root, "paris");
        std::vector<std::vector<double>> paras_aggr (update_paras.get_value());
        updateNodesParas(root, paras_aggr);
        if (husky::Context::get_global_tid() == 0){
        Node* test_node = search(root, "paris");
        husky::LOG_I << "after updating paris->para[0]: " << test_node->para[0];
        }


        if (husky::Context::get_global_tid() == 0 && m % 10 == 9){
            distance(root, "paris", 10);
            distance(root, "windows", 10);
        }

    }

    return;
}

// For testing with single thread
void train_single(int iter, Node* root, int batch){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    // aggregate the parameters and number of samples
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
 
    list_execute(document_list, {}, {}, [&](Document& doc) {
        if (doc.batch == batch){
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

                        if(context_node->id() == context_node->parent->lchild->id()){
                            tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                        }
                        else{
                        
                            tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                        }
                         
                        int location = context_node->parent->location;
                        // update node vector
                        for (int m = 0; m < vectorSize; m++){
                            updateWord[m] += tempValue*context_node->parent->para[m];
                        }

                        for (int m = 0; m < vectorSize; m++) {
                            context_node->parent->para[m] += tempValue * word_node->para[m];
                        }
                        context_node = context_node->parent;
                    }
                    for (int m = 0; m < vectorSize; m++){
                        word_node->para[m] += updateWord[m];
                    }
                }
                

            }
        
        }//endif
    });
    return;
}

// For testing how it works with multiple threads when only single thread is available
void train_imitate_mult(int iter, int count, Node* root){
    auto& document_list = husky::ObjListStore::get_objlist<Document>(1);
    // use document-list to train the node paras, including word vectors locally
    // aggregate the parameters and number of samples
    if (husky::Context::get_global_tid() == 0) {
         husky::LOG_I << "begin training";
    }
    husky::lib::Aggregator<std::vector<std::vector<double>>> update_paras(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)),
      [](std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        if(a.size() == b.size()){
            for (int j = 0; j < b.size(); j++) {
                if (a.at(j).size() == b.at(j).size()){
                    for (int k = 0; k < a.at(j).size(); k++){
                        a.at(j).at(k) = a.at(j).at(k) + b.at(j).at(k);
                    }
                }
            }
        }
      },
      [&](std::vector<std::vector<double>>& v) {
           v = std::move(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)));
      });
    std::vector<std::vector<double>> original_paras = getNodesParas(root);
    for(int a = 0; a < numBatch; a++){
        list_execute(document_list, {}, {}, [&](Document& doc) {
            if (doc.batch == a){
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

                            if(context_node->id() == context_node->parent->lchild->id()){
                                tempValue = learningRate * (1 - logistic_value(word_node->para, context_node->parent->para));

                            }
                            else{
                            
                                tempValue = learningRate * ( 0 - logistic_value(word_node->para, context_node->parent->para));
                            }
                             
                            int location = context_node->parent->location;
                            // update node vector
                            for (int m = 0; m < vectorSize; m++){
                                updateWord[m] += tempValue*context_node->parent->para[m];
                            }

                            for (int m = 0; m < vectorSize; m++) {
                                context_node->parent->para[m] += tempValue * word_node->para[m];
                            }
                            
                            context_node = context_node->parent;
                        }
                        for (int m = 0; m < vectorSize; m++){
                            word_node->para[m] += updateWord[m];
                        }
                    }
                    

                }
            
            }//endif
        });
        std::vector<std::vector<double>> paras = getNodesParas(root);
                
        update_paras.update_any([&](std::vector<std::vector<double>>& v){
            for(int k = 0; k < count; k++){
                for(int b = 0; b < vectorSize; b++){
                    v[k][b] += paras[k][b];
                }
            }
        });
        std::stack<Node*> nodes;
        updateNodesParasSingle(root, original_paras); 
    }
    husky::lib::AggregatorFactory::sync();
    std::vector<std::vector<double>> paras_aggr (update_paras.get_value());
    updateNodesParas(root, paras_aggr);

    return;
}




std::pair<int, Node*> buildHuffmanTree(auto wc_list){
    std::priority_queue<Node*, std::vector<Node*>, comparator> minHeap;
    srand (time(NULL));
    int id = 0;
    int count = wc_list.size();
    husky::lib::Aggregator<std::vector<std::vector<double>>> update_paras(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)),
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
           v = std::move(std::vector<std::vector<double>>(count, std::vector<double>(vectorSize, 0)));
      });

    auto& ac = husky::lib::AggregatorFactory::get_channel();

    if (husky::Context::get_global_tid() == 0){
        for (int i = 0; i < wc_list.size(); i++){
            if(isInteger(wc_list[i].first)){
                if (wc_list[i].first.size()>7){
                    count --;
                    continue;
                }
                id = std::max(id,std::stoi(wc_list[i].first) + 1);
            }
            Node* node = new Node(wc_list[i].first);
            node->count = wc_list[i].second;
            // para size: vectorSize
            // random initialization
            for (int j = 0; j < vectorSize; j++){
                node->para.push_back((double(rand())/RAND_MAX - 0.5) / vectorSize);
            }
            update_paras.update_any([&](std::vector<std::vector<double>>& v){
                for (int m = 0; m < vectorSize; m++) {
                    v[i][m] = node->para[m];
                }
            });
            minHeap.push(node);
        }
    }
    husky::lib::AggregatorFactory::sync();
    std::vector<std::vector<double>> paras_aggr(update_paras.get_value());
    if (husky::Context::get_global_tid() != 0){
        for (int i = 0; i < wc_list.size(); i++){
            if(isInteger(wc_list[i].first)){
                if (wc_list[i].first.size()>7){
                    count --;
                    continue;
                }
                id = std::max(id, std::stoi(wc_list[i].first) + 1);
            }
            Node* node = new Node(wc_list[i].first);
            node->count = wc_list[i].second;
            for (int j = 0; j < vectorSize; j++){
                node->para.push_back(paras_aggr[i][j]);
            }
            minHeap.push(node);
        }
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
    }
    Node* top_node = minHeap.top();
    minHeap.pop();
    // set location
    inOrderTraversal(top_node);
    std::pair<int, Node*> reValues = std::make_pair(count, top_node);
    return reValues;
    // Testing functions
    if (husky::Context::get_global_tid() == 0) {
        Node* word_node = search(top_node, "hippies");
        husky::LOG_I << "word: " << word_node->id() << "  para size: " << word_node->para.size() << " para[0,-1]: " << word_node->para[0] << word_node->para[vectorSize - 1];
    }
}



auto createVocab() {
    // term_list: store every term in the corpus
    auto& term_list = husky::ObjListStore::create_objlist<Term>();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "begin createVocab()";
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
    // Finish loading
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
    // Here, the word count has finished and wc_list is a list of (term, count).
    auto wc_list (wc.get_value());
 
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "wc_list size: "<< wc_list.size();
    }
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << wc_list.at(0).first;
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
    
    husky::lib::Aggregator<int> num_total_words(0,
        [](int& a, const int& b){ a += b; },
        0
    );

    auto& num_channel = husky::lib::AggregatorFactory::get_channel();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "before loading";
    }

    if (husky::Context::get_global_tid() == 0){ 
        // calculate the total number of words
        
        FILE * file;
        file = fopen (filename , "r");
        long fileSize;
        char * content;
        size_t result;
        // obtain file size:
        fseek (file , 0 , SEEK_END);
        fileSize = ftell (file);
        rewind (file);

        content = (char*) malloc (sizeof(char)*fileSize/numPartitions);
        for (int i = 0; i < numPartitions; i++){
            std::string id;
            id = std::to_string(i);
            Document doc(id);
            // copy the file into the buffer:
            result = fread (content,1,fileSize/numPartitions,file);
            boost::char_separator<char> sep(" \t\n.,()\'\":;!?<>[]-=|");
            std::string scontent(content);
            boost::tokenizer<boost::char_separator<char>> tok(scontent, sep);
            for (auto& w : tok) {
                doc.words.push_back(w);
                std::transform(doc.words.back().begin(), doc.words.back().end(), doc.words.back().begin(), ::tolower); 
                num_total_words.update(1);
            }
      
            doc.batch = rand() % numBatch;
            document_list.add_object(doc);
        }
        fclose(file);
        free(content);
    }
    globalize(document_list);
    husky::lib::AggregatorFactory::sync();
    list_execute(document_list, {}, {&num_term}, [&](Document& doc){
        for (std::string w : doc.words){
            num_term.push(1, w);
        }
    });
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << " after pushing ";
    }

    int maxCount = (int) num_total_words.get_value() * maxFrequency;
    list_execute(term_list, {&num_term}, {&ac}, [&](Term& t){
        t.count = num_term.get(t);
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
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << " after dropping list ";
    }

    auto wc_list (wc.get_value());
 
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << " after";
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
}


void word2vec() {
    if (husky::Context::get_global_tid() == 0) {
        ProfilerStart("/tmp/jiayi/a.prof");
    }
    //single machine

   /* 
    // auto wc_list = createVocab();
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
        distance(root, "graduate", 10);
        distance(root, "windows", 10);
        distance(root, "english", 10);
        
        learningRate = starting_learningRate * (1 - k / numIterations);
        if (learningRate < starting_learningRate * 0.0001){
            learningRate = starting_learningRate * 0.0001;
        }

        // train_single(k + 1, root, 1);
        // train_imitate_mult(k+1, numNodes, root);
        if (husky::Context::get_global_tid() == 0) {
            husky::LOG_I << "after iteration" << k;
        }

    }
    distance(root, "graduate", 10);
    distance(root, "windows", 10);
    distance(root, "english", 10);

    return;
    */
    //multiple machines
   
    // auto wc_list = createVocab();
    // auto wc_list = createVocabByFile("/data/jiayi/husky_github/husky/word2vec/trunk/text8");
    auto wc_list = createVocabByFile("/data/jiayi/data/enwik9");
    auto rePair = buildHuffmanTree(wc_list);
    int numNodes = rePair.first;
    Node* root = rePair.second;
    if (husky::Context::get_global_tid() == 0){
        husky::LOG_I << "before creating map";
    }
    std::unordered_map<std::string, Node*> map = createMap(root);
    if (husky::Context::get_global_tid() == 0){
        husky::LOG_I << "after creating map";
    }

    for (int k = 0; k < numIterations; k++){
        learningRate = starting_learningRate * (1 - k / numIterations);
        if (learningRate < starting_learningRate * 0.0001){
            learningRate = starting_learningRate * 0.0001;
        }

            train_v2(map ,numNodes, root);
            //test word vectors

        if (husky::Context::get_global_tid() == 0){
            husky::LOG_I << "after iteration" << k;
            distance(root, "graduate", 10);
            distance(root, "windows", 10);
            distance(root, "english", 10);

        }
        std::string w1 = "paris";
        std::string w2 = "france";
        std::string w3 = "italy";
        if (husky::Context::get_global_tid() <= 3){
            std::string result = test(w1, w2, w3, root);
            husky::LOG_I << "Result1: " << result;
        }
        w1 = "king";
        w2 = "man";
        w3 = "woman";
        if (husky::Context::get_global_tid() <= 3){
            std::string result = test(w1, w2, w3, root);
            husky::LOG_I << "Result2: " << result;
        }

        
    }
    std::string w1 = "paris";
    std::string w2 = "france";
    std::string w3 = "italy";
    if (husky::Context::get_global_tid() <= 3){
        husky::LOG_I << " after training";
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result1: " << result;
    }
    w1 = "king";
    w2 = "man";
    w3 = "woman";
    if (husky::Context::get_global_tid() <= 3){
        std::string result = test(w1, w2, w3, root);
        husky::LOG_I << "Result2: " << result;
    }

    return; 
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
