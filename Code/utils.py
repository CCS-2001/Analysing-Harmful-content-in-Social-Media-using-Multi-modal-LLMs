from datasets import load_dataset, Dataset
from toxigen import label_annotations
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json, os

'''
Function to retrieve data from dataset

Inputs:
    dataset_name (str): the name of the dataset
    category (str): the harmful content category of the dataset
    instruct (bool): whether to include instructions or not, default is True
    multimodal (bool): whether the dataset is multimodal, default is False

Returns:
    ds (Dataset): test set of the dataset
    fs_all (list): list of few shot prompts for the dataset
    cot_prompt (str): the chain of thought prompt for the dataset

'''
def retrieve_dataset(dataset_name, category, instruct=True, multimodal=False):
    fs_all = []
    instruction = f'###INSTRUCTION: Given the following text, classify it as either {category} (YES) or not (NO). \
If it is {category}, answer YES. If it is not {category}, answer NO. You must answer with only YES or NO.'    
    cot_prompt = None
    
    if dataset_name == 'OLID':
        dataset = load_dataset("christophsonntag/OLID")
        train_df = Dataset.to_pandas(dataset['train'])
        test_df = Dataset.to_pandas(dataset['test'])
        label_dict = {'NOT': 'NO',
                         'OFF': 'YES'}

        # extract text and label and rename them
        train_df = train_df.loc[:, ['tweet', 'subtask_a']]
        train_df.columns = ['text', 'label']
        df = test_df.loc[:, ['tweet', 'subtask_a']]
        df.columns = ['text', 'label']
        
        # change labels to 0 and 1
        df.loc[df['label'] == 'NOT', 'label'] = 0
        df.loc[df['label'] != 0, 'label'] = 1

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: @USER She should ask a few native Americans what their take on this is.
###RATIONALE: This statement is offensive because it reduces Native Americans to tokens in a conversation, generalizes their experiences, \
and suggests that their perspectives are only being invoked to serve an argument rather than out of genuine interest in their views. \
Additionally, the dismissive or patronizing tone can contribute to the offensiveness of the remark.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <@USER She should ask a few native Americans what their take on this is.> considered {category}?  
Rationale: This statement is offensive because it reduces Native Americans to tokens in a conversation, generalizes their experiences, \
and suggests that their perspectives are only being invoked to serve an argument rather than out of genuine interest in their views. \
Additionally, the dismissive or patronizing tone can contribute to the offensiveness of the remark.           
Answer: YES'''
    
    elif dataset_name == 'HateXplain':
        dataset = load_dataset('Datasets/hatexplain.py', trust_remote_code=True)
        train_df = Dataset.to_pandas(dataset['train'])
        df = Dataset.to_pandas(dataset['test'])
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # format text and labels of the dataset
        train_df['text'] = train_df['post_tokens'].apply(lambda x: ' '.join(x))
        train_df['label'] = train_df['annotators'].apply(lambda x: np.bincount(x['label']).argmax())
        df['text'] = df['post_tokens'].apply(lambda x: ' '.join(x))
        df['label'] = df['annotators'].apply(lambda x: np.bincount(x['label']).argmax())
        
        # extract only text and label and exclude the label that is not hate or normal 
        df = df.loc[:, ['text', 'label']]
        df = df[df['label'] != 2]
        train_df = train_df[train_df['label'] != 2]
        
        # flip the labels so 0 is NO and 1 is YES
        df['label'] = 1 - df['label']
        train_df['label'] = 1 - train_df['label']

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: in britain the prime minister a childless woman london mayor a moslem invader in command of london police a lesbian in command of london fire brigade a fat woman scotland first minister a childless woman
###RATIONALE: The text is hate speech because it unfairly and offensively targets individuals based on personal characteristics such as gender, \
religion, sexual orientation, and appearance, using those characteristics to demean and discredit them. This type of language fosters \
prejudice, discrimination, and social division by perpetuating harmful stereotypes and dehumanizing those mentioned.
###ANSWER: YES
'''
        else:
            cot_prompt = f'''Question: Is the following text <in britain the prime minister a childless woman london mayor a moslem invader in command of london police a lesbian in command of london fire brigade a fat woman scotland first minister a childless woman> considered {category}?  
Rationale: The text is hate speech because it unfairly and offensively targets individuals based on personal characteristics such as gender, \
religion, sexual orientation, and appearance, using those characteristics to demean and discredit them. This type of language fosters \
prejudice, discrimination, and social division by perpetuating harmful stereotypes and dehumanizing those mentioned.          
Answer: YES'''
        
    elif dataset_name == 'Toxigen':
        dataset = load_dataset("toxigen/toxigen-data", 'annotated')
        pd_train = pd.DataFrame(dataset["train"])
        pd_test = pd.DataFrame(dataset["test"])
        
        train_df = label_annotations(pd_train)
        label_dict = {0: 'NO',
                      1: 'YES'}

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(label_annotations(pd_test))

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: asians are trying to take over the world with their karate skills
###RATIONALE: The text is toxic because it spreads harmful racial stereotypes, promotes fear and division between ethnic groups, and frames an \
entire racial group as a global threat. Such language fosters xenophobia, perpetuates ignorance, and reinforces negative biases that can \
contribute to racism and social division.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <asians are trying to take over the world with their karate skills> considered {category}?  
Rationale: The text is toxic because it spreads harmful racial stereotypes, promotes fear and division between ethnic groups, and frames an \
entire racial group as a global threat. Such language fosters xenophobia, perpetuates ignorance, and reinforces negative biases that can \
contribute to racism and social division.       
Answer: YES'''
        
    elif dataset_name == 'Toraman hate speech':
        df = pd.read_csv('Datasets/Toraman_hate_speech.tsv', sep='\t')

        # extract texts, label and language and exclude labels that is not hate speech or normal
        df = df.loc[:, ['text', 'label', 'language']]
        main_df = df[df['label'] != 1]
        main_df.loc[main_df['label'] == 2, 'label'] = 1
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # split into English and Turkish
        df_eng = df[df['language'] == 1]
        df_tr = df[df['language'] == 0]
        df_eng = df_eng.loc[:, ['text', 'label']]
        df_tr = df_tr.loc[:, ['text', 'label']]
        
        # randomly sample 500 samples of normal and hate from English and Turkish
        df_eng_0 = df_eng[df_eng['label'] == 0].sample(n=500, random_state=42)
        df_eng_1 = df_eng[df_eng['label'] == 2].sample(n=500, random_state=42)
        
        df_tr_0 = df_tr[df_tr['label'] == 0].sample(n=500, random_state=42)
        df_tr_1 = df_tr[df_tr['label'] == 2].sample(n=500, random_state=42)
        
        # join and shuffle all the samples together
        df = pd.concat([df_eng_0, df_eng_1, df_tr_0, df_tr_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # change label from 2 to 1 
        df.loc[df['label'] == 2, 'label'] = 1

        # Merge main_df and df and use indicator to identify rows in both or only in main_df
        diff_df = pd.merge(main_df, df, how='left', indicator=True)
        
        # Filter out rows that are present in both to create train set
        train_df = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # repeat the same steps for train set
        train_df_eng = train_df[train_df['language'] == 1]
        train_df_tr = train_df[train_df['language'] == 0]
        train_df_eng_0 = train_df_eng[train_df_eng['label'] == 0]
        train_df_eng_1 = train_df_eng[train_df_eng['label'] == 1]
        train_df_tr_0 = train_df_tr[train_df_tr['label'] == 0]
        train_df_tr_1 = train_df_tr[train_df_tr['label'] == 1]
        
        # create list of few shot prompt
        train_df_list = [train_df_eng_0, train_df_tr_0, train_df_eng_1, train_df_tr_1]
        for train_df in train_df_list:
            if instruct:
                text = '###TEXT: ' + train_df.iloc[0]['text']
                answer = '###ANSWER: ' + label_dict[train_df.iloc[0]['label']]
                fs_all.append(f'{instruction}\n{text}\n{answer}')
            else:
                text = f"###QUESTION: Is the content of the following text <{train_df.iloc[0]['text']}> considered {category}?"
                answer = '###ANSWER: ' + label_dict[train_df.iloc[0]['label']]
                fs_all.append(f'{text}\n{answer}')

        # make into Dataset from pandas dataframe    
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: Transgender woman found dead in Christmas day slaying, advocacy group says https://t.co/fqtKOP14SE via @nbcnews \n\n\
This is the work of the christian taliban.
###RATIONALE: The text is hate speech because it unfairly and inaccurately targets Christians by linking them to an act of violence through \
the use of dehumanizing language, stereotyping, and baseless accusations. Such language fosters animosity and promotes prejudice and division among \
religious and social groups.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <Transgender woman found dead in Christmas day slaying, advocacy group says https://t.co/fqtKOP14SE via @nbcnews \n\n This is the work of the christian taliban.> considered {category}?  
Rationale: The text is hate speech because it unfairly and inaccurately targets Christians by linking them to an act of violence through \
the use of dehumanizing language, stereotyping, and baseless accusations. Such language fosters animosity and promotes prejudice and division among \
religious and social groups.     
Answer: YES'''
        
    elif dataset_name == 'Ejaz cyberbullying':
        df = pd.read_csv('Datasets/Ejaz_cyberbullying.csv')

        # extract and rename text aand label
        df = df.loc[:, ['Message', 'Label']] 
        df.columns = ['text', 'label']
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # exclude texts that are over 1000 characters
        df = df[df['text'].str.len() <= 1000]
        main_df = df.copy()

        # randomly sample 500 samples each of cyberbullying and normal
        df_0 = df[df['label'] == 0].sample(n=500, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=500, random_state=42)
        
        # join and shuffle all the samples together
        df = pd.concat([df_0, df_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Merge main_df and df and use indicator to identify rows in both or only in main_df
        diff_df = pd.merge(main_df, df, how='left', indicator=True)
        
        # Filter out rows that are present in both to create train set
        train_df = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: bye bye dear bajaj  i got some better work to do than watching ur funny bike launch
###RATIONALE: The text is cyberbullying due to its dismissive, mocking tone, public ridicule, and lack of constructive engagement. It is designed to belittle the target and can contribute to feelings of embarrassment or inadequacy, which are common effects of cyberbullying.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <bye bye dear bajaj  i got some better work to do than watching ur funny bike launch> considered {category}?  
Rationale: The text is cyberbullying due to its dismissive, mocking tone, public ridicule, and lack of constructive engagement. It is designed to belittle the target and can contribute to feelings of embarrassment or inadequacy, which are common effects of cyberbullying.  
Answer: YES'''
    

    elif dataset_name == 'OffensEval-TR 2020':
        dataset = load_dataset('Datasets/offenseval2020_tr.py', trust_remote_code=True)
        train_df = Dataset.to_pandas(dataset['train'])
        df = Dataset.to_pandas(dataset['test'])
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # extract and rename text and label columns
        train_df = train_df.loc[:, ['tweet', 'subtask_a']]
        df = df.loc[:, ['tweet', 'subtask_a']]
        train_df.columns = ['text', 'label']
        df.columns = ['text', 'label']

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: @USER @USER @USER Reis bu ülkenin Devlet Başkanı ve Ak parti genel başkanı saygı duyacaksınız hergele
###RATIONALE: The text is offensive due to its use of insulting language ("hergele"), its aggressive and authoritarian tone in demanding respect, \
and its dismissal of differing opinions. The overall effect is one of intimidation, disrespect, and hostility, which contributes to an offensive \
and divisive message.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <@USER @USER @USER Reis bu ülkenin Devlet Başkanı ve Ak parti genel başkanı saygı duyacaksınız hergele> considered {category}?  
Rationale: The text is offensive due to its use of insulting language ("hergele"), its aggressive and authoritarian tone in demanding respect, \
and its dismissal of differing opinions. The overall effect is one of intimidation, disrespect, and hostility, which contributes to an offensive \
and divisive message.
Answer: YES'''
    
    elif dataset_name == 'SOSNet cyberbullying':
        df = pd.read_csv('Datasets/SOSNet_cyberbullying.csv')
        df.columns = ['text', 'label']
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # change labels to 0 and 1
        df.loc[df['label'] == 'not_cyberbullying', 'label'] = 0
        df.loc[df['label'] != 0, 'label'] = 1
        main_df = df.copy()
        
        # sample 500 random samples of cyberbullying and normal
        df_0 = df[df['label'] == 0].sample(n=500, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=500, random_state=42)

        # join and shuffle the sample together
        df = pd.concat([df_0, df_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Merge main_df and df and use indicator to identify rows in both or only in main_df
        diff_df = pd.merge(main_df, df, how='left', indicator=True)
        
        # Filter out rows that are present in both to make train set
        train_df = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: You never saw any celebrity say anything like this for Obama: B Maher Incest Rape 'Joke' S Colbert Gay \
'joke' K Griffin beheading 'joke'
###RATIONALE: The text is a form of cyberbullying because it publicly targets specific individuals, shames them for past controversial actions, \
and encourages a negative narrative about them. The use of sensitive and harmful topics further amplifies the damage, while the overall intent \
appears to be humiliation rather than constructive criticism. By perpetuating negativity and encouraging others to join in on the shaming, \
the text contributes to a toxic online environment.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <You never saw any celebrity say anything like this for Obama: B Maher Incest Rape 'Joke' S Colbert Gay 'joke' K Griffin beheading 'joke'> considered {category}?  
Rationale: The text is a form of cyberbullying because it publicly targets specific individuals, shames them for past controversial actions, \
and encourages a negative narrative about them. The use of sensitive and harmful topics further amplifies the damage, while the overall intent \
appears to be humiliation rather than constructive criticism. By perpetuating negativity and encouraging others to join in on the shaming, \
the text contributes to a toxic online environment.
Answer: YES'''
    
    elif dataset_name == 'LLM-JP Toxicity':
        ds = load_dataset('p1atdev/LLM-jp-Toxicity-Dataset', split='train')
        df = Dataset.to_pandas(ds)
        label_dict = {0: 'NO',
                     1: 'YES'}
        
        # extract text and label columns
        df = df.loc[:, ['text', 'label']]

        # change labels to 0 and 1
        df.loc[df['label'] == 'toxic', 'label'] = 1
        df.loc[df['label'] != 1, 'label'] = 0

        # filter out samples with over 1000 samples
        df = df[df['text'].str.len() <= 1000]
        main_df = df.copy()
        
        # randomly sample 200 samples of toxic and normal
        df_0 = df[df['label'] == 0].sample(n=200, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=200, random_state=42)
        
        # join and shuffle the samples together
        df = pd.concat([df_0, df_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Merge main_df and df and use indicator to identify rows in both or only in main_df
        diff_df = pd.merge(main_df, df, how='left', indicator=True)
        
        # Filter out rows that are present in both to make train set
        train_df = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        train_df['text_length'] = train_df['text'].apply(len)

        # Sort the DataFrame by 'text_length' column from shortest to longest
        train_df = train_df.sort_values(by='text_length')
        
        # Drop the 'text_length' column if no longer needed
        train_df = train_df.drop(columns=['text_length'])
    
        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

        # define chain of thought prompt
        if instruct:
            cot_prompt = '''###TEXT: # エリート変態女-尿道電気責め・連続絶頂\u3000※サンプル動画あり\n\nトップページ ＞ SM・拷問・調教 作品 一覧 ＞ エリート変態女-尿道電気責め・連続絶頂\u3000※サンプル動画あり\n\n※なかなかの良作です。 SMといっても、流血など苦手なのでこれくらいの方がありがたい。とはいえ、責めは激しく女優さんもしっかり痛がっています。電気って痛いんですね。 普段刺激されないであろう尿道なので、痛みもひとしおなのかも知れません。 SM初級者にも是非オススメです。
###RATIONALE: The text is toxic because it promotes abuse, violence, and disrespect toward others through the use of violent, dehumanizing, and sexually explicit language.
###ANSWER: YES'''
        else:
            cot_prompt = f'''Question: Is the following text <# エリート変態女-尿道電気責め・連続絶頂\u3000※サンプル動画あり\n\nトップページ ＞ SM・拷問・調教 作品 一覧 ＞ エリート変態女-尿道電気責め・連続絶頂\u3000※サンプル動画あり\n\n※なかなかの良作です。 SMといっても、流血など苦手なのでこれくらいの方がありがたい。とはいえ、責めは激しく女優さんもしっかり痛がっています。電気って痛いんですね。 普段刺激されないであろう尿道なので、痛みもひとしおなのかも知れません。 SM初級者にも是非オススメです。> considered {category}?  
Rationale: The text is toxic because it promotes abuse, violence, and disrespect toward others through the use of violent, dehumanizing, and sexually explicit language.
Answer: YES'''

    elif dataset_name == 'MMHS150K':
        # load the names of each image file
        with open('Datasets/MMHS150K/test_ids.txt') as file:
            lines = file.readlines()
            test_ids = [x.strip() for x in lines]

        with open('Datasets/MMHS150K/MMHS150K_GT.json') as file:
            data = json.load(file)
      
        image_names = []
        text = []
        labels = []
        # go though each image file, check if it exists and extract the corresponding text file for each image
        for test in test_ids:
            try:
                with open('Datasets/MMHS150K/img_txt/'+test+'.json') as file:
                    img_text = json.load(file)
                    text.append(img_text['img_text'])
            except:
                text.append('')
                            
            image_names.append(test + '.jpg')
            label = data[test]['labels']
            if label.count(0) >= 2:
                label = 0
            else:
                label = 1
                
            labels.append(label)

        # create pandas dataframe with image file name, text and label
        df = pd.DataFrame()
        df['image_name'] = image_names
        df['text'] = text
        df['label'] = labels
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

    elif dataset_name == 'MultiOFF':
        df = pd.read_csv('Datasets/MultiOFF/Testing_meme_dataset.csv')

        # rename columns and labels
        df.columns = ['image_name', 'text', 'label']
        df.loc[df['label'] == 'Non-offensiv', 'label'] = 0
        df.loc[df['label'] != 0, 'label'] = 1

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)

    elif dataset_name == 'MultiToxic':
        df = pd.read_csv('Datasets/MultiToxic/MultiToxic.csv')

        # extract and rename columns
        df = df.loc[:, ['image_path', 'text', 'is_toxic']]
        df.columns = ['image_name', 'text', 'label']
        
        # retain the image file name omly
        df['image_name'] = df['image_name'].str.split('/').str[-1]
        
        # randomly sample 300 samples of toxic and normal
        df_0 = df[df['label'] == 0].sample(n=300, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=300, random_state=42)

        # join and shuffle the samples together
        df = pd.concat([df_0, df_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)
    
    elif dataset_name == 'MultiBully':
        df = pd.read_excel('Datasets/MultiBully/MultiBully.xlsx')

        # remove empty rows
        df = df.dropna(subset=['Img_Name', 'Img_Text_Label'])
        
        # extract and rename columns
        df = df.loc[:, ['Img_Name', 'Img_Text', 'Img_Text_Label']]
        df.columns = ['image_name', 'text', 'label']
        
        drop = []
        # check each image file if it exists or is corrupted
        for img in df['image_name']:
            file_path = 'Datasets/MultiBully/Images/' + img
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    drop.append(img)
            except:
                drop.append(img)
        
        # remove entries of images that needs to be removed from df
        df = df[~df['image_name'].isin(drop)]
        
        # change the labels to 0 and 1
        df.loc[df['label'] == 'Nonbully', 'label'] = 0
        df.loc[df['label'] != 0, 'label'] = 1
        
        # randomly sample 500 samples of cyberbullying and normal
        df_0 = df[df['label'] == 0].sample(n=500, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=500, random_state=42)

        # join and shuffle all the samples together
        df = pd.concat([df_0, df_1])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # make into Dataset from pandas dataframe
        ds = Dataset.from_pandas(df)
        
    else:
        raise('No dataset provided')

    # create list of few shot prompts
    if dataset_name != 'Toraman hate speech' and not multimodal:
        for i in range(5):
            if instruct:
                text = '###TEXT: ' + train_df.iloc[i]['text']
                answer = '###ANSWER: ' + label_dict[train_df.iloc[i]['label']]
                fs_all.append(f'{instruction}\n{text}\n{answer}')
            else:
                text = f"Question: Is the following text <{train_df.iloc[i]['text']}> considered as {category}?"
                answer = 'Answer: ' + label_dict[train_df.iloc[i]['label']]
                fs_all.append(f'{text}\n{answer}')


    return ds, fs_all, cot_prompt


'''
Function to display metric scores and confusion matrix

Inputs:
    true_labels (list): list of true labels
    pred_labels (list): list of predicted labels
    labels (list): list of display labels
    model (str): the name of the language model
    dataset_name (str): the name of the dataset
    num_shot (int): the number of shots
    cot (bool): whether it is chain of thought prompting, default is False

Returns:
    None

'''
def display_results(true_labels, pred_labels, labels, model, dataset_name, num_shot, cot=False):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, pred_labels)*100
    
    # Print the metrics
    if cot:
        print('LLM:', model, 'Dataset:', dataset_name, 'chain of thought')
    else:
        print('LLM:', model, 'Dataset:', dataset_name, num_shot, 'shot')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.2f}')
    
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()