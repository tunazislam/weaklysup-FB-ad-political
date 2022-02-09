import numpy
import random
from random import shuffle
from numpy import array
from Embedder import Embedder
import torch
import torch.optim as optim
import time
import pickle


class NodeInputData:

    def __init__(self, batch_size=30):

        with open('data.pickle', 'rb') as in_file:
            [self.graph, self.id2name, self.name2id, self.stance_start, self.stance_end, self.issue_start, self.issue_end, self.ad_start, self.ad_end,\
             self.funding_entity_start, self.funding_entity_end, self.issue_unigram_start, self.issue_unigram_end, self.personal_unigram_start,\
             self.personal_unigram_end, self.policy_unigram_start, self.policy_unigram_end, self.indicator_label_start, self.indicator_label_end, \
             self.annotated_ads, self.annotated_funding_entities, self.ad2tokenized_text, self.weights_matrix, self.id2text, self.entity_indicators, \
             self.left_entity_indicators, self.right_entity_indicators, self.ad2annotated_stance, self.ad2annotated_issue,\
             self.ad2personal_policy_annotation]=pickle.load(in_file, encoding="bytes")

        self.all_stances=set([i for i in range(self.stance_start, self.stance_end+1)])
        self.all_issues=set([i for i in range(self.issue_start, self.issue_end+1)])
        self.all_ads=set([i for i in range(self.ad_start, self.ad_end+1)])
        self.all_funding_entities=set([i for i in range(self.funding_entity_start, self.funding_entity_end+1)])
        self.all_issue_unigrams=set([i for i in range(self.issue_unigram_start, self.issue_unigram_end+1)])
        self.all_personal_unigrams=set([i for i in range(self.personal_unigram_start, self.personal_unigram_end+1)])
        self.all_policy_unigrams=set([i for i in range(self.policy_unigram_start, self.policy_unigram_end+1)])
        self.all_indicator_labels=set([i for i in range(self.indicator_label_start, self.indicator_label_end+1)])
        


        self.ad2stance_adj_pos={}
        self.ad2funding_entity_adj_pos={}
        self.ad2personal_unigram_adj_pos={}
        self.ad2policy_unigram_adj_pos={}
        self.ad2issue_unigram_adj_pos={}
        self.funding_entity2stance_adj_pos={}
        self.issue_unigram2issue_adj_pos={}
        self.issue2indicator_label_adj_pos={}
        self.personal_unigram2indicator_label_adj_pos={}
        self.policy_unigram2indicator_label_adj_pos={}

        self.ad2stance_adj_neg = {}
        self.ad2funding_entity_adj_neg = {}
        self.ad2personal_unigram_adj_neg = {}
        self.ad2policy_unigram_adj_neg = {}
        self.ad2issue_unigram_adj_neg = {}
        self.funding_entity2stance_adj_neg = {}
        self.issue_unigram2issue_adj_neg = {}
        self.issue2indicator_label_adj_neg = {}
        self.personal_unigram2indicator_label_adj_neg = {}
        self.policy_unigram2indicator_label_adj_neg = {}

        self.all_nodes_to_train=None

        self.batch_size=batch_size


    def get_nodes(self, training_graph):

        #print (len(training_graph))
        
        for ad in self.all_ads:
            adjacency_list=training_graph[ad]

            if len(adjacency_list & self.all_stances)>0:
                self.ad2stance_adj_pos[ad]=adjacency_list & self.all_stances
                self.ad2stance_adj_neg[ad] = self.all_stances - self.ad2stance_adj_pos[ad]
            if len(adjacency_list & self.all_funding_entities)>0:
                self.ad2funding_entity_adj_pos[ad]=adjacency_list & self.all_funding_entities
                self.ad2funding_entity_adj_neg[ad] = self.all_funding_entities - self.ad2funding_entity_adj_pos[ad]
            if len(adjacency_list & self.all_personal_unigrams)>0:
                self.ad2personal_unigram_adj_pos[ad]=adjacency_list & self.all_personal_unigrams
                self.ad2personal_unigram_adj_neg[ad] = self.all_personal_unigrams - self.ad2personal_unigram_adj_pos[ad]
            if len(adjacency_list & self.all_policy_unigrams)>0:
                self.ad2policy_unigram_adj_pos[ad]=adjacency_list & self.all_policy_unigrams
                self.ad2policy_unigram_adj_neg[ad] = self.all_policy_unigrams - self.ad2policy_unigram_adj_pos[ad]
            if len(adjacency_list & self.all_issue_unigrams)>0:
                self.ad2issue_unigram_adj_pos[ad]=adjacency_list & self.all_issue_unigrams
                self.ad2issue_unigram_adj_neg[ad] = self.all_issue_unigrams - self.ad2issue_unigram_adj_pos[ad]

        for funding_entity in self.all_funding_entities:
            adjacency_list=training_graph[funding_entity]

            if len(adjacency_list & self.all_stances)>0:
                self.funding_entity2stance_adj_pos[funding_entity]=adjacency_list & self.all_stances
                self.funding_entity2stance_adj_neg[funding_entity] = self.all_stances - self.funding_entity2stance_adj_pos[funding_entity]

        for iu in self.all_issue_unigrams:
            
            adjacency_list = training_graph[iu]
            #print (adjacency_list)
            
            if len(adjacency_list & self.all_issues) > 0:
                self.issue_unigram2issue_adj_pos[iu] = adjacency_list & self.all_issues
                self.issue_unigram2issue_adj_neg[iu] = self.all_issues - self.issue_unigram2issue_adj_pos[iu]

        for pu in self.all_personal_unigrams:
            adjacency_list = training_graph[pu]

            if len(adjacency_list & self.all_indicator_labels) > 0:
                self.personal_unigram2indicator_label_adj_pos[pu] = adjacency_list & self.all_indicator_labels
                self.personal_unigram2indicator_label_adj_neg[pu] = self.all_indicator_labels - self.personal_unigram2indicator_label_adj_pos[pu] - set([self.name2id['$issue']])

        for pu in self.all_policy_unigrams:
            adjacency_list = training_graph[pu]

            if len(adjacency_list & self.all_indicator_labels) > 0:
                self.policy_unigram2indicator_label_adj_pos[pu] = adjacency_list & self.all_indicator_labels
                self.policy_unigram2indicator_label_adj_neg[pu] = self.all_indicator_labels - self.policy_unigram2indicator_label_adj_pos[pu] - set([self.name2id['$issue']])

        for issue in self.all_issues:
            adjacency_list = training_graph[issue]

            if len(adjacency_list & self.all_indicator_labels) > 0:
                self.issue2indicator_label_adj_pos[issue] = adjacency_list & self.all_indicator_labels
                self.issue2indicator_label_adj_neg[issue] = self.all_indicator_labels - self.issue2indicator_label_adj_pos[issue]


        
        
        self.all_nodes_to_train=[i for i in training_graph]

        batches=[]
        
        discarded_nodes=0
        shuffle(self.all_nodes_to_train)
        
        #print (len(self.all_nodes_to_train))
        j=0
        while j<len(self.all_nodes_to_train):
            batch_input = {'ad2stance': [], 'ad2funding_entity': [], 'ad2personal_unigram': [], 'ad2policy_unigram': [], 'ad2issue_unigram': [],\
                           'funding_entity2stance': [], 'issue_unigram2issue': [], 'personal_unigram2indicator_label': [],\
                           'policy_unigram2indicator_label': [], 'issue2indicator_label': []}
            batch_gold = {'ad2stance': [], 'ad2funding_entity': [], 'ad2personal_unigram': [], 'ad2policy_unigram': [], 'ad2issue_unigram': [],\
                           'funding_entity2stance': [], 'issue_unigram2issue': [], 'personal_unigram2indicator_label': [],\
                           'policy_unigram2indicator_label': [], 'issue2indicator_label': []}
            batch_neg = {'ad2stance': [], 'ad2funding_entity': [], 'ad2personal_unigram': [], 'ad2policy_unigram': [], 'ad2issue_unigram': [],\
                           'funding_entity2stance': [], 'issue_unigram2issue': [], 'personal_unigram2indicator_label': [],\
                           'policy_unigram2indicator_label': [], 'issue2indicator_label': []}

            text_nodes=[]
            
            if j+self.batch_size<=len(self.all_nodes_to_train)-1:
                nodes_to_train=self.all_nodes_to_train[j:j+self.batch_size]
            else:
                nodes_to_train = self.all_nodes_to_train[j:]

            if len(nodes_to_train)==0:
                continue


            for node in nodes_to_train:
                if node in self.all_ads:
                    
                    
                    text_nodes.append(node)

                    if node in self.ad2stance_adj_pos:
                        #print ("ad2st", node)
                        batch_input['ad2stance']+=[node for i in range(0,len(self.ad2stance_adj_pos[node]))]
                        batch_gold['ad2stance']+=list(self.ad2stance_adj_pos[node])
                        batch_neg['ad2stance']+=[random.sample(self.ad2stance_adj_neg[node], 2) for i in range(0, len(self.ad2stance_adj_pos[node]))]

                    if node in self.ad2funding_entity_adj_pos:
                        #print ('ad2fe', node)
                        batch_input['ad2funding_entity']+=[node for i in range(0,len(self.ad2funding_entity_adj_pos[node]))]
                        batch_gold['ad2funding_entity']+=list(self.ad2funding_entity_adj_pos[node])
                        batch_neg['ad2funding_entity']+=[random.sample(self.ad2funding_entity_adj_neg[node], 5) for i in range(0, len(self.ad2funding_entity_adj_pos[node]))]

                    if node in self.ad2personal_unigram_adj_pos:
                        #print ('ad2per', node)
                        batch_input['ad2personal_unigram'] += [node for i in range(0, len(self.ad2personal_unigram_adj_pos[node]))]
                        batch_gold['ad2personal_unigram'] += list(self.ad2personal_unigram_adj_pos[node])
                        batch_neg['ad2personal_unigram'] += [random.sample(self.ad2personal_unigram_adj_neg[node], 5) for i in range(0, len(self.ad2personal_unigram_adj_pos[node]))]

                    if node in self.ad2policy_unigram_adj_pos:
                        #print ('ad2pol', node)
                        batch_input['ad2policy_unigram'] += [node for i in range(0, len(self.ad2policy_unigram_adj_pos[node]))]
                        batch_gold['ad2policy_unigram'] += list(self.ad2policy_unigram_adj_pos[node])
                        batch_neg['ad2policy_unigram'] += [random.sample(self.ad2policy_unigram_adj_neg[node], 5) for i in range(0, len(self.ad2policy_unigram_adj_pos[node]))]

                    if node in self.ad2issue_unigram_adj_pos:
                        #print ('ad2is', node)
                        batch_input['ad2issue_unigram'] += [node for i in range(0, len(self.ad2issue_unigram_adj_pos[node]))]
                        batch_gold['ad2issue_unigram'] += list(self.ad2issue_unigram_adj_pos[node])
                        batch_neg['ad2issue_unigram'] += [random.sample(self.ad2issue_unigram_adj_neg[node], 5) for i in range(0, len(self.ad2issue_unigram_adj_pos[node]))]


                if node in self.all_funding_entities:
                    if node in self.funding_entity2stance_adj_pos:
                        #print ('fe2st', node)
                        batch_input['funding_entity2stance']+=[node for i in range(0,len(self.funding_entity2stance_adj_pos[node]))]
                        batch_gold['funding_entity2stance']+=list(self.funding_entity2stance_adj_pos[node])
                        batch_neg['funding_entity2stance']+=[random.sample(self.funding_entity2stance_adj_neg[node], 2) for i in range(0, len(self.funding_entity2stance_adj_pos[node]))]


                if node in self.all_issue_unigrams:
                    if node in self.issue_unigram2issue_adj_pos:
                        #print ('isu2is', node)
                        batch_input['issue_unigram2issue']+=[node for i in range(0,len(self.issue_unigram2issue_adj_pos[node]))]
                        batch_gold['issue_unigram2issue']+=list(self.issue_unigram2issue_adj_pos[node])
                        batch_neg['issue_unigram2issue']+=[random.sample(self.issue_unigram2issue_adj_neg[node], 5) for i in range(0, len(self.issue_unigram2issue_adj_pos[node]))]

                if node in self.all_personal_unigrams:
                    if node in self.personal_unigram2indicator_label_adj_pos:
                        #print ('per2in', node)
                        batch_input['personal_unigram2indicator_label']+=[node for i in range(0,len(self.personal_unigram2indicator_label_adj_pos[node]))]
                        batch_gold['personal_unigram2indicator_label']+=list(self.personal_unigram2indicator_label_adj_pos[node])
                        batch_neg['personal_unigram2indicator_label']+=[random.sample(self.personal_unigram2indicator_label_adj_neg[node], 1) for i in range(0, len(self.personal_unigram2indicator_label_adj_pos[node]))]

                if node in self.all_policy_unigrams:
                    if node in self.policy_unigram2indicator_label_adj_pos:
                        #print ('pol2in', node)
                        batch_input['policy_unigram2indicator_label'] += [node for i in range(0, len(self.policy_unigram2indicator_label_adj_pos[node]))]
                        batch_gold['policy_unigram2indicator_label'] += list(self.policy_unigram2indicator_label_adj_pos[node])
                        batch_neg['policy_unigram2indicator_label'] += [random.sample(self.policy_unigram2indicator_label_adj_neg[node], 1) for i in range(0, len(self.policy_unigram2indicator_label_adj_pos[node]))]

                if node in self.all_issues:
                    if node in self.issue2indicator_label_adj_pos:
                        #print ('is2in', node)
                        batch_input['issue2indicator_label'] += [node for i in range(0, len(self.issue2indicator_label_adj_pos[node]))]
                        batch_gold['issue2indicator_label'] += list(self.issue2indicator_label_adj_pos[node])
                        batch_neg['issue2indicator_label'] += [random.sample(self.issue2indicator_label_adj_neg[node], 2) for i in range(0, len(self.issue2indicator_label_adj_pos[node]))]


            for k in batch_input:
                batch_input[k] = array(batch_input[k])
            for k in batch_gold:
                '''
                if k=="text2label" or k=="followee2label" or k=="node2label" or k=='frame2label' or k=='hashtag2label' or k=='bigram2label' or k=='trigram2label':
                    batch_gold[k] = array(batch_gold[k])
                else:
                    batch_gold[k] = array([batch_gold[k]]).transpose()
                '''
                batch_gold[k] = array([batch_gold[k]]).transpose()

            for k in batch_neg:
                batch_neg[k] = array(batch_neg[k])

            j=j+self.batch_size
            batches.append([batch_input, batch_gold, batch_neg, text_nodes])
        
        self.batches=batches
        
        print ("Discarded: %d nodes."%(discarded_nodes))
        #return
        
        '''
        i=0
        for batch in self.batches:
            print (batch)
            i+=1
            if i==10:
                break
        '''
    def save_embeddings(self, model):

        output_dir='../../scratch/fb_ads_data/output/embeddings/'
        
        f=open(output_dir+"stance.embeddings","w")
        stance_embd=model.stance_embeddings.weight.cpu().data.numpy()
        stance_embd=stance_embd.tolist()
        for i in range(0,len(stance_embd)):
            id=i+self.stance_start
            name=self.id2name[id]
            f.write(str(name))
            embd=stance_embd[i]
            for j in range(len(embd)):
                if j==0:
                    f.write("\t"+str(embd[j]))
                else:
                    f.write(" "+str(embd[j]))
            f.write("\n")

        f = open(output_dir + "personal_unigram.embeddings", "w")
        personal_unigram_embd = model.personal_unigram_embeddings.weight.cpu().data.numpy()
        personal_unigram_embd = personal_unigram_embd.tolist()
        for i in range(0, len(personal_unigram_embd)):
            id = i + self.personal_unigram_start
            name = self.id2name[id]
            f.write(str(name))
            embd = personal_unigram_embd[i]
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")

        f = open(output_dir + "policy_unigram.embeddings", "w")
        policy_unigram_embd = model.policy_unigram_embeddings.weight.cpu().data.numpy()
        policy_unigram_embd = policy_unigram_embd.tolist()
        for i in range(0, len(policy_unigram_embd)):
            id = i + self.policy_unigram_start
            name = self.id2name[id]
            f.write(str(name))
            embd = policy_unigram_embd[i]
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")

        f = open(output_dir + "issue_unigram.embeddings", "w")
        issue_unigram_embd = model.issue_unigram_embeddings.weight.cpu().data.numpy()
        issue_unigram_embd = issue_unigram_embd.tolist()
        for i in range(0, len(issue_unigram_embd)):
            id = i + self.issue_unigram_start
            name = self.id2name[id]
            f.write(str(name))
            embd = issue_unigram_embd[i]
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")

        f = open(output_dir + "issue.embeddings", "w")
        issue_embd = model.issue_embeddings.weight.cpu().data.numpy()
        issue_embd = issue_embd.tolist()
        for i in range(0, len(issue_embd)):
            id = i + self.issue_start
            name = self.id2name[id]
            f.write(str(name))
            embd = issue_embd[i]
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")

        f = open(output_dir + "indicator_label.embeddings", "w")
        indicator_label_embd = model.indicator_label_embeddings.weight.cpu().data.numpy()
        indicator_label_embd = indicator_label_embd.tolist()
        for i in range(0, len(indicator_label_embd)):
            id = i + self.indicator_label_start
            name = self.id2name[id]
            f.write(str(name))
            embd = indicator_label_embd[i]
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")

        f = open(output_dir + "ad.embeddings", "w")
        ads = [k for k in range(self.ad_start, self.ad_end + 1)]
        for d in ads:
            embd = model.ad_embeddings([d])
            embd = embd[0]
            embd = embd.cpu().data.numpy()
            embd = embd.tolist()
            name = self.id2name[d]
            f.write(str(name))
            for j in range(len(embd)):
                if j == 0:
                    f.write("\t" + str(embd[j]))
                else:
                    f.write(" " + str(embd[j]))
            f.write("\n")
        f.close()



    def GetTestBatch(self,type):
            docs=[]
            labels=[]

            considered_docs=None

            if type=="train":
                considered_docs=shared
            elif type=="test":
                considered_docs=not_shared
            elif type=="all":
                considered_docs = shared+not_shared

            for doc in considered_docs:
                docs.append(doc)
                labels.append(self.all_text2label[doc])
        
            batch=[array(docs),labels]
            return batch

        
        
def run():
    global cuda_available

    NID=NodeInputData(batch_size=100)
    print ("Creating batches...")
    NID.get_nodes(NID.graph)
    print ("Batches Created!")
    cuda_available = torch.cuda.is_available()
    
    #return
    print ("Initializing the model...")

    model = Embedder(cuda_available=cuda_available)
    model1 = Embedder(cuda_available=cuda_available)
    model2 = Embedder(cuda_available=cuda_available)
   
    print ("Model Initiated!")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print ("Total Batches: %d"%len(NID.batches))
    
    for name, param in model1.named_parameters():
        if param.requires_grad:
            print (name)

    min_loss=10000000
    no_up=0
    model.train()
    for epoch in range(0, 100): #initiate number of epoches
        start=time.time()
        average_loss=0
        count=0
        for batch in NID.batches:
            #print (batch)
            optimizer.zero_grad()
            l,loss=model(batch)
            
            #print (l)
            
            if l==0:
                continue
            
            l.backward()
            optimizer.step()
            #average_loss+=l.data[0]
            average_loss+=l.data
           
        end=time.time()
        print ("Total Time for epoch: %d: %lf"%(epoch, float(end-start)))
        print ("Loss at epoch %d = %f"%(epoch, average_loss))
        
        if average_loss<min_loss:
            min_loss=average_loss
            model1.load_state_dict(model.state_dict())

            no_up=0
        else:
            no_up+=1
        
        '''
        model.eval()
        model.predict_stance()
        model.predict_issue()
        model.predict_policy_personal()
        model.train()
        '''
        if no_up==10:
            break

    NID.save_embeddings(model1)
    
    
    model.eval()
    model.predict_stance()
    model.predict_issue()
    model.predict_policy_personal()
    model.eval()
    
    model1.eval()
    model1.predict_stance()
    model1.predict_issue()
    model1.predict_policy_personal()
    model1.eval()
    
    
    ad2stance=model1.predict_stance_all_ad()
    ad2issue=model1.predict_issue_all_ad()
    fe2stance=model1.predict_stance_all_funding_entity()
    
    output_dir='../../scratch/fb_ads_data/output/other/'
    with open(output_dir+'predicted_stances_issues.pickle', 'wb') as out_file:
        pickle.dump([ad2stance, ad2issue, fe2stance], out_file)
        
    output_dir='../../scratch/fb_ads_data/output/model/'
    torch.save(model1.state_dict(), output_dir+"moodel.m")
    
    
    model2.load_state_dict(torch.load(output_dir+"moodel.m"))
    model2.eval()
    model2.predict_stance()
    model2.predict_issue()
    model2.predict_policy_personal()
    model2.eval()
    
    
    return model1


def dot_product(x, y):
    x=numpy.array(x)
    y=numpy.array(y)
    return numpy.dot(x,y)


    
if __name__ == '__main__':
    
    random.seed(1234)
    torch.manual_seed(1234)
    numpy.random.seed(1234)

    model=run()
