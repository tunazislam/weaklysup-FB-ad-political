import math
from BLSTM import *
import random


class Embedder(torch.nn.Module):
    def __init__(self, cuda_available=True):
        super(Embedder, self).__init__()

        with open('data.pickle', 'rb') as in_file:
            [self.graph, self.id2name, self.name2id, self.stance_start, self.stance_end, self.issue_start, self.issue_end, self.ad_start, self.ad_end,\
             self.funding_entity_start, self.funding_entity_end, self.issue_unigram_start, self.issue_unigram_end, self.personal_unigram_start,\
             self.personal_unigram_end, self.policy_unigram_start, self.policy_unigram_end, self.indicator_label_start, self.indicator_label_end, \
             self.annotated_ads, self.annotated_funding_entities, self.ad2tokenized_text, self.weights_matrix, self.id2text, self.entity_indicators, \
             self.left_entity_indicators, self.right_entity_indicators, self.ad2annotated_stance, self.ad2annotated_issue,\
             self.ad2personal_policy_annotation]=pickle.load(in_file, encoding="bytes")

        self.stance_count=self.stance_end-self.stance_start+1
        self.issue_count=self.issue_end-self.issue_start+1
        self.ad_count=self.ad_end-self.ad_start+1
        self.funding_entity_count=self.funding_entity_end-self.funding_entity_start+1
        self.issue_unigram_count=self.issue_unigram_end-self.issue_unigram_start+1
        self.policy_unigram_count=self.policy_unigram_end-self.policy_unigram_start+1
        self.personal_unigram_count=self.personal_unigram_end-self.personal_unigram_start+1
        self.indicator_label_count=self.indicator_label_end-self.indicator_label_start+1

        self.cuda_available=cuda_available
        self.embedding_size = 300


        self.scale = 1.0 / math.sqrt(self.embedding_size)
        
        self.embed_rng = np.random.RandomState(1)

        if self.cuda_available:
            self.stance_embeddings = nn.Embedding(self.stance_count, self.embedding_size).cuda()
            self.funding_entity_embeddings = nn.Embedding(self.funding_entity_count, self.embedding_size).cuda()
            self.issue_embeddings = nn.Embedding(self.issue_count, self.embedding_size).cuda()
            self.issue_unigram_embeddings = nn.Embedding(self.issue_unigram_count, self.embedding_size).cuda()
            self.policy_unigram_embeddings = nn.Embedding(self.policy_unigram_count, self.embedding_size).cuda()
            self.personal_unigram_embeddings = nn.Embedding(self.personal_unigram_count, self.embedding_size).cuda()
            self.indicator_label_embeddings = nn.Embedding(self.indicator_label_count, self.embedding_size).cuda()

        else:
            self.stance_embeddings = nn.Embedding(self.stance_count, self.embedding_size)
            self.funding_entity_embeddings = nn.Embedding(self.funding_entity_count, self.embedding_size)
            self.issue_embeddings = nn.Embedding(self.issue_count, self.embedding_size)
            self.issue_unigram_embeddings = nn.Embedding(self.issue_unigram_count, self.embedding_size)
            self.policy_unigram_embeddings = nn.Embedding(self.policy_unigram_count, self.embedding_size)
            self.personal_unigram_embeddings = nn.Embedding(self.personal_unigram_count, self.embedding_size)
            self.indicator_label_embeddings = nn.Embedding(self.indicator_label_count, self.embedding_size)


        stance_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.stance_count, self.embedding_size))
        funding_entity_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.funding_entity_count, self.embedding_size))
        issue_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.issue_count, self.embedding_size))
        issue_unigram_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.issue_unigram_count, self.embedding_size))
        policy_unigram_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.policy_unigram_count, self.embedding_size))
        personal_unigram_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.personal_unigram_count, self.embedding_size))
        indicator_label_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.indicator_label_count, self.embedding_size))

        self.stance_embeddings.weight.data.copy_(torch.from_numpy(stance_embed_values))
        self.funding_entity_embeddings.weight.data.copy_(torch.from_numpy(funding_entity_embed_values))
        self.issue_embeddings.weight.data.copy_(torch.from_numpy(issue_embed_values))
        self.issue_unigram_embeddings.weight.data.copy_(torch.from_numpy(issue_unigram_embed_values))
        self.policy_unigram_embeddings.weight.data.copy_(torch.from_numpy(policy_unigram_embed_values))
        self.personal_unigram_embeddings.weight.data.copy_(torch.from_numpy(personal_unigram_embed_values))
        self.indicator_label_embeddings.weight.data.copy_(torch.from_numpy(indicator_label_embed_values))

        if self.cuda_available:
            self.ad_embeddings = BLSTM().cuda()
        else:
            self.ad_embeddings = BLSTM()


        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')

        self.aspects = ['ad2stance', 'ad2funding_entity', 'ad2personal_unigram', 'ad2policy_unigram', 'ad2issue_unigram',\
                        'funding_entity2stance',\
                        'issue_unigram2issue',\
                        'personal_unigram2indicator_label',\
                        'policy_unigram2indicator_label',\
                        'issue2indicator_label']

    def decision(self, probability):
        return random.random() < probability

    def forward(self, batch):

        batch_input_index = batch[0]
        batch_gold_index = batch[1]
        batch_negs_index = batch[2]
        text_nodes = batch[3]

        if len(text_nodes) > 0:

            textid2index = {}
            for i in range(0, len(text_nodes)):
                textid2index[text_nodes[i]] = i
            if self.cuda_available:
                '''
                hlstm only
                '''
                output_doc_embeddings = self.ad_embeddings(text_nodes).cuda()
            else:
                '''
                hlstm only
                '''
                output_doc_embeddings = self.ad_embeddings(text_nodes)


        embedding_size = self.embedding_size
        batch_target_index = {}
        batch_input = {}
        batch_target = {}
        input_embed = {}
        target_embed = {}
        sim_score = {}
        loss_all = {}

        '''
        all_weights = {}
        all_weights['softmax_W'] = {}
        all_weights['softmax_B'] = {}
        '''

        loss = 0
        for aspect in self.aspects:
            if (len(batch_gold_index[aspect]) == 0):
                continue

            
            if (aspect == 'ad2stance'):
                target_embeddings = self.stance_embeddings
            elif (aspect == 'ad2funding_entity'):
                target_embeddings = self.funding_entity_embeddings
            elif (aspect == 'ad2personal_unigram'):
                continue
                target_embeddings = self.personal_unigram_embeddings
            elif (aspect == 'ad2policy_unigram'):
                continue
                target_embeddings = self.policy_unigram_embeddings
            elif (aspect == 'ad2issue_unigram'):
                target_embeddings = self.issue_unigram_embeddings
            elif (aspect == 'funding_entity2stance'):
                target_embeddings = self.stance_embeddings
            elif (aspect == 'issue_unigram2issue'):
                target_embeddings = self.issue_embeddings
            elif (aspect == 'personal_unigram2indicator_label'):
                continue
                target_embeddings = self.indicator_label_embeddings
            elif (aspect == 'policy_unigram2indicator_label'):
                continue
                target_embeddings = self.indicator_label_embeddings
            elif (aspect == 'issue2indicator_label'):
                continue
                target_embeddings = self.indicator_label_embeddings
            else:
                continue


            if (aspect == 'ad2stance'):
                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.stance_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.stance_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.stance_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.stance_start))), 1)

            elif (aspect == 'ad2funding_entity'):
                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.funding_entity_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.funding_entity_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.funding_entity_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.funding_entity_start))), 1)


            elif (aspect == 'ad2personal_unigram'):
                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.personal_unigram_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.personal_unigram_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.personal_unigram_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.personal_unigram_start))), 1)

            elif (aspect == 'ad2policy_unigram'):
                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.policy_unigram_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.policy_unigram_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.policy_unigram_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.policy_unigram_start))), 1)

            elif (aspect == 'ad2issue_unigram'):
                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.issue_unigram_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.issue_unigram_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.issue_unigram_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.issue_unigram_start))), 1)

            elif (aspect == 'funding_entity2stance'):
                if self.cuda_available:
                    batch_input[aspect] = self.funding_entity_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.funding_entity_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.stance_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.stance_start))), 1)
                else:
                    batch_input[aspect] = self.funding_entity_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.funding_entity_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.stance_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.stance_start))), 1)

            elif (aspect == 'issue_unigram2issue'):
                if self.cuda_available:
                    batch_input[aspect] = self.issue_unigram_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.issue_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.issue_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.issue_start))), 1)
                else:
                    batch_input[aspect] = self.issue_unigram_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.issue_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.issue_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.issue_start))), 1)

            elif (aspect == 'personal_unigram2indicator_label'):
                if self.cuda_available:
                    batch_input[aspect] = self.personal_unigram_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.personal_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)
                else:
                    batch_input[aspect] = self.personal_unigram_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.personal_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)

            elif (aspect == 'policy_unigram2indicator_label'):
                if self.cuda_available:
                    batch_input[aspect] = self.policy_unigram_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.policy_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)
                else:
                    batch_input[aspect] = self.policy_unigram_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.policy_unigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)

            elif (aspect == 'issue2indicator_label'):
                if self.cuda_available:
                    batch_input[aspect] = self.issue_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.issue_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)
                else:
                    batch_input[aspect] = self.issue_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.issue_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.indicator_label_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.indicator_label_start))), 1)




            #if aspect in ['ad2stance', 'funding_entity2stance', 'personal_unigram2indicator_label', 'policy_unigram2indicator_label', 'issue2indicator_label']:
            if aspect in ['ad2stance', 'funding_entity2stance', 'issue2indicator_label']:
                example_size = 2 + 1
            elif aspect in ['personal_unigram2indicator_label', 'policy_unigram2indicator_label']:
                example_size = 1 + 1
            else:
                example_size = 5 + 1


            batch_target[aspect] = target_embeddings(batch_target_index[aspect]).view((-1, example_size, self.embedding_size))

            dropout = nn.Dropout(p=0.7)
            input_layers = [dropout(batch_input[aspect])]
            input_embed[aspect] = input_layers[-1]

            target_layers = [dropout(batch_target[aspect])]
            target_embed[aspect] = target_layers[-1]

            sim_score[aspect] = torch.bmm(
                target_embed[aspect],
                input_embed[aspect].view(-1, embedding_size, 1)).view(-1, example_size)

            if self.cuda_available:
                target = Variable(torch.cuda.LongTensor(batch_input[aspect].size(0)).zero_())
            else:
                target = Variable(torch.LongTensor(batch_input[aspect].size(0)).zero_())

            self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')

            loss_all[aspect] = self.CrossEntropyLoss(sim_score[aspect], target)

            loss_all[aspect] = torch.sum(loss_all[aspect])

            rate = 1.0

            loss += loss_all[aspect] * rate

        return loss, loss_all

    def chunks(self, l, n):
        n = max(1, n)
        return (l[i:i + n] for i in range(0, len(l), n))

    def dot_product(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return np.dot(x, y)

    def predict_stance(self):
        batch_input_index = [id for id in self.ad2annotated_stance]
        batch_gold=[self.ad2annotated_stance[id] for id in self.ad2annotated_stance]

        stance_embd = self.stance_embeddings.weight.cpu().data.numpy()
        stance_embd = stance_embd.tolist()
        batch_input_index = self.chunks(batch_input_index, 30)
        doc_embds = []
        for chunk in batch_input_index:
            embd = self.ad_embeddings(chunk)
            embd = (embd.cpu().data.numpy()).tolist()
            doc_embds += embd

        predictions = []
        for i in range(0, len(doc_embds)):
            d_embd = doc_embds[i]
            scores = []
            for l in range(0, len(stance_embd)):
                l_embd = stance_embd[l]
                scores.append(self.dot_product(d_embd, l_embd))

            predictions.append([scores.index(max(scores))])


        correct=0
        for i in range(len(predictions)):
            if len(set(predictions[i]) & set(batch_gold[i]))>0:
                correct+=1


        print (correct/float(len(predictions)))
        
        
    def predict_issue(self):
        batch_input_index = [id for id in self.ad2annotated_issue]
        batch_gold = [self.ad2annotated_issue[id] for id in self.ad2annotated_issue]

        issue_embd = self.issue_embeddings.weight.cpu().data.numpy()
        issue_embd = issue_embd.tolist()
        batch_input_index = self.chunks(batch_input_index, 30)
        doc_embds = []
        for chunk in batch_input_index:
            embd = self.ad_embeddings(chunk)
            embd = (embd.cpu().data.numpy()).tolist()
            doc_embds += embd

        predictions = []
        for i in range(0, len(doc_embds)):
            d_embd = doc_embds[i]
            scores = []
            for l in range(0, len(issue_embd)):
                l_embd = issue_embd[l]
                scores.append(self.dot_product(d_embd, l_embd))

            predictions.append([scores.index(max(scores))+self.issue_start])

        correct = 0
        for i in range(len(predictions)):
            if len(set(predictions[i]) & set(batch_gold[i])) > 0:
                correct += 1

        print (correct / float(len(predictions)))
        
    def predict_policy_personal(self):
        batch_input_index = [id for id in self.ad2personal_policy_annotation]
        batch_gold = [self.ad2personal_policy_annotation[id] for id in self.ad2personal_policy_annotation]

        indicator_embd = self.indicator_label_embeddings.weight.cpu().data.numpy()
        indicator_embd = indicator_embd.tolist()
        batch_input_index = self.chunks(batch_input_index, 30)
        doc_embds = []
        for chunk in batch_input_index:
            embd = self.ad_embeddings(chunk)
            embd = (embd.cpu().data.numpy()).tolist()
            doc_embds += embd

        predictions = []
        for i in range(0, len(doc_embds)):
            d_embd = doc_embds[i]
            scores = []
            for l in range(1, len(indicator_embd)):
                l_embd = indicator_embd[l]
                scores.append(self.dot_product(d_embd, l_embd))

            predictions.append(scores.index(max(scores))+self.indicator_label_start+1)

        correct = 0
        corrects=[]
        for i in range(len(predictions)):
            if predictions[i]==batch_gold[i]:
                correct += 1
                corrects.append(predictions[i])
        
        
        for i in set(corrects):
            print (i, corrects.count(i))

        print (correct / float(len(predictions)))
        
        
    def predict_stance_all_funding_entity(self):
        funding_entity_embds=self.funding_entity_embeddings.weight.cpu().data.numpy()
        funding_entity_embds=funding_entity_embds.tolist()
            
        stance_embd = self.stance_embeddings.weight.cpu().data.numpy()
        stance_embd = stance_embd.tolist()


        fe2stance={}
        for i in range(0, len(funding_entity_embds)):
            fe_embd = funding_entity_embds[i]
            scores = []
            for l in range(0, len(stance_embd)):
                l_embd = stance_embd[l]
                scores.append(self.dot_product(fe_embd, l_embd))

            stance2score={}
            for j in range(len(scores)):
                stance2score[self.id2name[j+self.stance_start]]=scores[j]

            sorted_stance={k: v for k, v in sorted(stance2score.items(), key=lambda item: item[1], reverse=True)}

            fe2stance[self.id2name[i+self.funding_entity_start]]=sorted_stance

        return fe2stance

    def predict_stance_all_ad(self):
        _batch_input_index = [id for id in range(self.ad_start, self.ad_end + 1)]
        # batch_gold = [self.ad2annotated_issue[id] for id in self.ad2annotated_issue]

        stance_embd = self.stance_embeddings.weight.cpu().data.numpy()
        stance_embd = stance_embd.tolist()
        batch_input_index = self.chunks(_batch_input_index, 30)
        ad_embds = []
        for chunk in batch_input_index:
            embd = self.ad_embeddings(chunk)
            embd = (embd.cpu().data.numpy()).tolist()
            ad_embds += embd

        ad2stance = {}
        for i in range(0, len(ad_embds)):
            ad_embd = ad_embds[i]
            scores = []
            for l in range(0, len(stance_embd)):
                l_embd = stance_embd[l]
                scores.append(self.dot_product(ad_embd, l_embd))

            stance2score = {}
            for j in range(len(scores)):
                stance2score[self.id2name[j + self.stance_start]] = scores[j]

            sorted_stance = {k: v for k, v in sorted(stance2score.items(), key=lambda item: item[1], reverse=True)}

            ad2stance[self.id2name[_batch_input_index[i]]] = sorted_stance

        return ad2stance


    def predict_issue_all_ad(self):
        _batch_input_index = [id for id in range(self.ad_start, self.ad_end + 1)]
        # batch_gold = [self.ad2annotated_issue[id] for id in self.ad2annotated_issue]

        issue_embd = self.issue_embeddings.weight.cpu().data.numpy()
        issue_embd = issue_embd.tolist()
        batch_input_index = self.chunks(_batch_input_index, 30)
        ad_embds = []
        for chunk in batch_input_index:
            embd = self.ad_embeddings(chunk)
            embd = (embd.cpu().data.numpy()).tolist()
            ad_embds += embd

        ad2issue = {}
        for i in range(0, len(ad_embds)):
            ad_embd = ad_embds[i]
            scores = []
            for l in range(0, len(issue_embd)):
                l_embd = issue_embd[l]
                scores.append(self.dot_product(ad_embd, l_embd))

            issue2score = {}
            for j in range(len(scores)):
                issue2score[self.id2name[j + self.issue_start]] = scores[j]

            sorted_issue = {k: v for k, v in sorted(issue2score.items(), key=lambda item: item[1], reverse=True)}

            ad2issue[self.id2name[_batch_input_index[i]]] = sorted_issue

        return ad2issue


if __name__ == '__main__':
    print ("Dummy Main Function")
