from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

import soundfile as sf
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

import io
#torch.cuda.set_device(1)
#print(torch.cuda.current_device())


class Wav2VecExperimentRunner:
    def __init__(self, cuda, model_name):
        self.CUDA = cuda
        self.MODEL_NAME = model_name
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # self.prepare_model()


    def prepare_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_NAME).to(self.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.MODEL_NAME)
        self.extracto= AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    # def load_and_preprocess_audio(self, audio_path):
    #     # Load and preprocess audio here
    #     waveform, sample_rate = sf.read(audio_path)
    #     return waveform, sample_rate

    def load_data(self):
        waveform, sample_rate = sf.read('common_voice_sample.wav')
        return waveform, sample_rate


    # def prepare_data(self):

    #     test,_ = self.load_data()
    #     self.test = [test]

    def prepare_data(self):
        df=pd.read_csv('common_voice_sample_phonemes.csv')
        self.test=df
        # print('data: {}'.format(self.test))

    # def extract_features(self, audio):
    #     # inputs = self.tokenizer(audio, return_tensors="pt", padding="longest")
    #     # inputs = inputs.input_values.to(self.device)
    #     # with torch.no_grad():
    #     #     features = self.model(inputs).last_hidden_state
    #     return features



    def map_phones_to_features(self, features, audio_sample_rate, phonemes_df):
        """
        Map phonemes to audio features.

        :param features: Extracted features from wav2vec.
        :param audio_sample_rate: Sample rate of the audio.
        :param phonemes_df: DataFrame with columns ['phoneme', 'start_time', 'end_time'].
        :return: List of tuples (phoneme, feature_vector).
        """
        feature_vectors = []
        feature_frame_rate = features.shape[1] / audio_sample_rate  # Calculate feature frame rate

        for _, row in phonemes_df.iterrows():
            phoneme = row['phoneme']
            start_time = row['start_time']
            end_time = row['end_time']

            # Calculate the corresponding feature indices
            start_index = int(start_time * feature_frame_rate)
            end_index = int(end_time * feature_frame_rate)

            # Extract the corresponding feature vector
            if start_index < features.shape[1] and end_index <= features.shape[1]:
                feature_vector = features[:, start_index:end_index].mean(dim=1)  # Average features over the time window
                feature_vectors.append((phoneme, feature_vector))

        return feature_vectors



    def get_prediction_softmax(self, waveform, next_phoneme_start):
        # next_phoneme_start=0
        # Process the audio input with the model
        # waveform, sample_rate = sf.read('/mnt/Data/raghav/shapley_residuals_llm/final/data/'+audio_input['wav_file_name'])

        logits = self.model(waveform).logits
        prediction=logits.softmax(dim=-1)
        # print(logits)

        # Apply softmax from the next phoneme position onwards
        # if self.SOFTMAX:
        # prediction = logits[0, next_phoneme_start:, :].softmax(dim=-1)
        # else:
        #     prediction = logits[:, next_phoneme_start:, :]

        # Return the numpy array if not using CUDA, else return the tensor
        if not self.CUDA:
            return prediction.detach().numpy()
        else:
            return prediction
    def pad_tensor_to_target(self,tensor, target_size):
        padding_size = target_size - tensor.size(1)
        # Pad only the second dimension (left and right)
        return F.pad(tensor, (0, 0, 0, padding_size), 'constant', 0)
    def interaction_value_di(self, row, PhonemeA,PhonemeB,PhonemeA_start,PhonemeB_start,PhonemeA_end,PhonemeB_end,PhonemeA_dur,PhonemeB_dur):
        print(PhonemeA, PhonemeB)

        # token1, token2 = tokens

        # if self.MODEL_NAME == 'facebook/wav2vec2-base-960hs':
        #     token_next = 0
        # else:
        #     raise Exception("Not Implemented")
        waveform, sample_rate_og = sf.read('common_voice_sample.wav')
        print("------------------------")
        print(waveform.size, sample_rate_og)
        print(waveform, sample_rate_og)
        # audio = AudioSegment.from_file('/mnt/Data/raghav/shapley_residuals_llm/final/data/'+row['wav_file_name'])
        input_values = self.tokenizer(waveform, return_tensors="pt", padding='max_length',max_length=399).input_values.to(self.device)

        token_next=0
        AB = self.get_prediction_softmax(input_values, token_next)
        # print('AB: {}'.format(AB))
        # input_values_t1 = input_values.clone()

        # Calculate start and end samples
        start_sample = int(PhonemeA_start * sample_rate_og)
        end_sample = int(PhonemeA_end * sample_rate_og)
        waveform_A=np.copy(waveform)
        waveform_A[start_sample:end_sample] = 0

        # Read only the desired portion of the file
        # waveform2, sample_rate = sf.read('/mnt/Data/raghav/shapley_residuals_llm/final/data/'+row['wav_file_name'], start=start_sample, stop=end_sample)
        input_values2 = self.tokenizer(waveform_A, return_tensors="pt",padding='max_length',max_length=399).input_values.to(self.device)
        A = self.get_prediction_softmax(input_values2, token_next)
        # print('A: {}'.format(A))

        start_sample = int(PhonemeB_start * sample_rate_og)
        end_sample = int(PhonemeB_end * sample_rate_og)

        waveform_B=np.copy(waveform)
        waveform_B[start_sample:end_sample] = 0

        # Read only the desired portion of the file
        # waveform2, sample_rate = sf.read('/mnt/Data/raghav/shapley_residuals_llm/final/data/'+row['wav_file_name'], start=start_sample, stop=end_sample)
        input_values3 = self.tokenizer(waveform_B, return_tensors="pt",padding='max_length',max_length=399).input_values.to(self.device)
        B = self.get_prediction_softmax(input_values3, token_next)
        # print('B: {}'.format(B))


        start_sample = int(PhonemeA_start * sample_rate_og)
        end_sample = int(PhonemeA_end * sample_rate_og)
        waveform_phi=np.copy(waveform)
        waveform_phi[start_sample:end_sample] = 0
        start_sample = int(PhonemeB_start * sample_rate_og)
        end_sample = int(PhonemeB_end * sample_rate_og)
        waveform_phi[start_sample:end_sample]=0

        # waveform3, sample_rate = sf.read('/mnt/Data/raghav/shapley_residuals_llm/final/data/'+row['wav_file_name'], start=start_sample, stop=end_sample)
        input_values4 = self.tokenizer(waveform_phi, return_tensors="pt",padding='max_length',max_length=399).input_values.to(self.device)
        Phi = self.get_prediction_softmax(input_values4, token_next)
        target_size = 399  # Largest size in the second dimension
        A = self.pad_tensor_to_target(A, target_size).squeeze(0)
        B = self.pad_tensor_to_target(B, target_size).squeeze(0)
        AB = self.pad_tensor_to_target(AB, target_size).squeeze(0)
        Phi = self.pad_tensor_to_target(Phi, target_size).squeeze(0)
        # print(A.shape)
        # print('***********')
        # print(B.shape)
        # print('***********')
        # print(AB.shape)
        # print('***********')
        # print(Phi.shape)
        # print('***********')
        num =  AB- A - B + Phi
        print(num.shape)
        # print(num)
        print('***********')


        # num = torch.linalg.norm(num, dim=-1).cpu()
        # den = torch.linalg.norm(AB, dim=-1).cpu()
        # norm_num = torch.linalg.norm(num, dim=-1)
        # norm_AB = torch.linalg.norm(AB, dim=-1)
        num = torch.linalg.norm(num,2)
        den = torch.linalg.norm(AB,2)

        ans = num/den
        ans=ans.detach().cpu().numpy()
        # ans=torch.divide(torch.linalg.norm(num, ord=2, axis=-1).cpu(),torch.linalg.norm(AB,ord=2, axis=-1).cpu())
        print(ans)
        return ans
        # print(num/den)

        # return num/den









        # X_t1[0, token1] = self.tokenizer.pad_token_id
        # A = self.get_prediction_softmax(X_t1, token_next)
        
        # X_t2 = X.clone()
        # X_t2[0,token1] = self.tokenizer.pad_token_id
        # B = self.get_prediction_softmax(X_t2, token_next)

        # X_t12 = X.clone()
        # X_t12[0,token2] = self.tokenizer.pad_token_id
        # X_t12[0,token1] = self.tokenizer.pad_token_id
        # phi = self.get_prediction_softmax(X_t12, token_next)
        # # print(AB, A, B, phi)
        # val = AB - A - B + phi
        

        # if self.METHOD == 105:
        #     val = AB - A - B + phi
        #     val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
        #     res_list = [(1, val.detach(), token_next)]
        #     val = AB - A - B + phi
        #     val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB - phi, dim=1)).cpu()
        #     res_list.append((2, val.detach(), token_next))
        #     val = AB - A - B + phi
        #     val = torch.linalg.norm(val, dim=1).cpu()
        #     res_list.append((3, val.detach(), token_next))
        #     val = torch.divide(torch.linalg.norm(AB - A - B, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
        #     res_list.append((4, val.detach(), token_next))
        #     return res_list

        # assert False

        # if self.METHOD == 1:
        #     val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
        #     return val.detach(), token_next
        # elif self.METHOD == 2:
        #     val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB - phi, dim=1)).cpu()
        #     return val.detach(), token_next
        # elif self.METHOD == 3:
        #     val = torch.linalg.norm(val, dim=1).cpu()
        #     return val.detach(), token_next
        # elif self.method == 4:
        #     val = torch.divide(torch.linalg.norm(AB - A - B, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
        #     return val.detach(), token_next
    def calculate_interaction(self, row, row_number):
        interactions = []

        phoneme_list=row['phoneme']
        phoneme_start_list=row['phoneme_start']
        phoneme_end_list =row['phoneme_end']
        phoneme_duration_list =row['phoneme_duration']


        for i in range(len(phoneme_list) - 1):
            if i==0:
                continue
            print("Phoneme:", phoneme_list[i], "to", phoneme_list[i + 1])
            print("Start Time:", phoneme_start_list[i], "to", phoneme_start_list[i + 1])
            print("End Time:", phoneme_end_list[i], "to", phoneme_end_list[i + 1])
            print("Duration:", phoneme_duration_list[i], "to", phoneme_duration_list[i + 1])
            shap_res=self.interaction_value_di(row,phoneme_list[i],phoneme_list[i + 1],phoneme_start_list[i],phoneme_start_list[i + 1],phoneme_end_list[i],phoneme_end_list[i + 1],phoneme_duration_list[i],phoneme_duration_list[i+1])

            interaction_data = {
                "Row Number": row['wav_file_name'],
                "Phoneme From": phoneme_list[i],
                "Phoneme To": phoneme_list[i + 1],
                "Start Time From": phoneme_start_list[i],
                "Start Time To": phoneme_start_list[i + 1],
                "End Time From": phoneme_end_list[i],
                "End Time To": phoneme_end_list[i + 1],
                "Duration From": phoneme_duration_list[i],
                "Duration To": phoneme_duration_list[i + 1],
                # Add other relevant data here
            }
            interaction_data["Shap Res"] = shap_res
            interactions.append(interaction_data)
        print(interactions)
        return interactions





        # print(self.get_prediction_softmax(row,0))

        # encoded_len = len(encoded_row)
        # for j in range( min(self.SEQ_LEN, encoded_len)):
        #     probability = 0.05
        #     if random.random() < probability: 
        #         for k in range(j+1, min(self.SEQ_LEN, encoded_len, j+9)):
        #             if j+k >= encoded_len:
        #                 continue
        #             if encoded_row[j] == self.tokenizer.unk_token_id or encoded_row[k] == self.tokenizer.unk_token_id:
        #                 continue

        #             og = encoded_row.clone()
        #             og = og.reshape(1, -1)
        #             iv = self.interaction_value_di(og, [j, k])
        #             interactions.append([iv, abs((j-k)), row_number, j, k])
                
        # return interactions



    def run_avg_interactions(self, suffix=''):
        average_distance = []
        mwes = []
        print(self.test)
        # for row_number, row in tqdm(enumerate(self.test), total=len(self.test)):
        # self.test=self.test.groupby(['wav_file_name', 'txt_file_name', 'textgrid_name'])
        # self.test = self.test.agg({
        #     'phoneme': lambda x: list(x),
        #     'phoneme_start': lambda x: list(x),
        #     'phoneme_end': lambda x: list(x),
        #     'phoneme_duration': lambda x: list(x)
        # }).reset_index()
        # self.test = self.test.rename(columns = {'start' : 'phoneme_start', 'end': 'phoneme_end'})
        # self.test = self.test.agg({'phoneme': lambda x : list(x), 'phoneme_start': lambda x : list(x), 'phoneme_end': lambda x : list(x)})
        self.test = pd.DataFrame([{
            "phoneme": self.test["phoneme"].tolist(),
            "phoneme_start": self.test["start"].tolist(),
            "phoneme_end": self.test["end"].tolist(),
            "phoneme_duration": (self.test["end"] - self.test["start"]).tolist()
        }])

        print(self.test)

        # self.test=self.test.sample(n=5, random_state=42)
        results = []
        for index, row in self.test.iterrows():
            print('INDEX: {}:'.format(index))
            # print('chech: {} {}'.format(index,row))
            # if row['wav_file_name'] not in ['common_voice_en_38487410.wav', 'common_voice_en_38487412.wav', 'common_voice_en_38487548.wav']:
            #     continue
            print(row)

            results.extend(self.calculate_interaction(row, index))
        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv('interactions3.csv', index=False)
            # print(self.get_prediction_softmax(row,0))


            # encoded_row =  self.tokenizer(row, padding=False, is_split_into_words=True, truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt')
            # g = {}
            # for ix, el in enumerate(encoded_row.word_ids()):
            #     if el is not None:
            #         if el not in g:
            #             g[el] = []
            #         g[el].append(ix)
            # mwe = [row_number, list(g.values())]
            # encoded_row = encoded_row.input_ids[0]
            # if self.CUDA:
            #     encoded_row = encoded_row.cuda()

            # average_distance.extend(self.calculate_interaction(encoded_row, row_number))
            # mwes.append(mwe)
            # if row_number % 1000 == 0:
            #     print(len(average_distance))
            #     pickle.dump(average_distance, open(f'avg_{self.MODEL_NAME}{suffix}{self.LANGUAGE}.pkl','wb'))
            #     pickle.dump(mwes, open(f'mwe_{self.MODEL_NAME}{suffix}{self.LANGUAGE}.pkl','wb'))

    def run_experiment(self, mwe=True, avg=True, suffix=''):
        self.prepare_data()
        self.prepare_model()
        self.run_avg_interactions(suffix=suffix)

if __name__  == '__main__':
    Wav2VecExperimentRunner(cuda=False, model_name = 'facebook/wav2vec2-base-960h').run_experiment(suffix='100')
        # Main experiment logic
