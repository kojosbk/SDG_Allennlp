
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import base64
from  PIL import Image
from streamlit_option_menu import option_menu
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from allennlp import pretrained
import matplotlib.pyplot as plt
# from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from allennlp.predictors.predictor import Predictor

###################################

from functionforDownloadButtons import download_button

###################################

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# set_png_as_page_bg('resources/imgs/newbg.png')
# set_bg_hack('resources/imgs/MISSION STATEMENT (4).png')
logo = Image.open('resources\imgs\sdg-un-banner.jpg')
st.sidebar.image(logo, use_column_width=True)
# Data Loading
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("resources\imgs\Copy of Updated Presentation.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# App declaration
def main():
    
    st.sidebar.title("Pages")
    title_SO = """
    <div style="background-color:#eebd8a;padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
    <h3 style="color:black;text-align:center;">UNITED NATIONS SUSTAINABLE DEVELOPMENT GOAL\nText comprehension</h3>
    """
    st.markdown(title_SO, unsafe_allow_html=True)
    page_options = ["About App / Web page info","3 Functionality Tabs","About us","Contact us",]

    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == "About App / Web page info":
        with st.sidebar:
            choose = option_menu("About App", ["-Motive behind App","-Summary of the functions"],
                                icons=['tropical-storm', 'kanban'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#f1f2f6"},
                "icon": {"color": "#eebd8a", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#585858"},		
                    }
                    )
        if choose == "-Motive behind App":
                title_SOm = """
                <div style="padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
                <h3 style="color:black;text-align:center;">A study-based project that uses Shell as a foundation for proof of concept. 
                \nThe overall goal of the app is to investigate the possibility of utilising data science to gather 
                and analyse information about a particular organisation, corporation, or nation
                 to determine how well they are performing in relation to the Sustainable Development Goals of the United Nations.</h3>
                """
                title_SOmh = """
                <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
                <h3 style="color:black;text-align:center;">Motive behind App"""     
                st.markdown(title_SOmh, unsafe_allow_html=True)           
                st.markdown(title_SOm, unsafe_allow_html=True)
        if choose == "-Summary of the functions":
                title_SOsf = """
                <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
                <h3 style="color:black;text-align:center;">Summary of the functions"""     
                st.markdown(title_SOsf, unsafe_allow_html=True)    
                title_SOsft = """
                <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
                <h3 style="color:black;text-align:center;"> ---------------------------------------------- 
                \nText Summrization - Without affecting the article's substance, the text summary tool draws out the most crucial details from a paragraph.
                \nQuestion & Answer - Users can ask questions that will be addressed in the articles using the question-answer (Q&A) feature.
                \nName Entity Recognition - By using the named entity recognition (NER) capability, it is possible to determine whether any entities have ever been connected to Shell Oil Company. Currently, we are only focusing on businesses and locations that are mentioned in the articles.""" 
                st.markdown(title_SOsft, unsafe_allow_html=True)                        
    if page_selection == "3 Functionality Tabs":
        with st.sidebar:
            choose = option_menu("Functionality Tabs", ["-Text summarisation","-Q n A","-Entity recognition"],
                                icons=['tropical-storm', 'tree', 'kanban'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#f1f2f6"},
                "icon": {"color": "#eebd8a", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#585858"},		
                    }
                    )
        if choose == "-Text summarisation":
            title_SOts = """
            <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">Text summarisation"""     
            st.markdown(title_SOts, unsafe_allow_html=True) 

            model = "BART"

            if model == 'BART':
                _num_beams = 4
                _no_repeat_ngram_size = 3
                _length_penalty = 1
                _min_length = 12
                _max_length = 128
                _early_stopping = True
            else:
                _num_beams = 4
                _no_repeat_ngram_size = 3
                _length_penalty = 2
                _min_length = 30
                _max_length = 200
                _early_stopping = True


            _num_beams = 4
            _no_repeat_ngram_size = 3
            _length_penalty = 2

            col1, col2, col3 = st.columns(3)
            _min_length = col1.number_input("Minimum length of summarization", value=_min_length)
            _max_length = col3.number_input("Maximum length summarization", value=_max_length)
            _early_stopping = 1#col3.number_input("early_stopping", value=_early_stopping)

            text = st.text_area('Text Input')


            def run_model(input_text):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                if model == "BART":
                    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
                    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
                    input_text = str(input_text)
                    input_text = ' '.join(input_text.split())
                    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
                    summary_ids = bart_model.generate(input_tokenized,
                                                    num_beams=_num_beams,
                                                    no_repeat_ngram_size=_no_repeat_ngram_size,
                                                    length_penalty=_length_penalty,
                                                    min_length=_min_length,
                                                    max_length=_max_length,
                                                    early_stopping=_early_stopping)

                    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                    st.write('Summary')
                    st.success(output[0])

                else:
                    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
                    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
                    input_text = str(input_text).replace('\n', '')
                    input_text = ' '.join(input_text.split())
                    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
                    summary_task = torch.tensor([[21603, 10]]).to(device)
                    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
                    summary_ids = t5_model.generate(input_tokenized,
                                                    num_beams=_num_beams,
                                                    no_repeat_ngram_size=_no_repeat_ngram_size,
                                                    length_penalty=_length_penalty,
                                                    min_length=_min_length,
                                                    max_length=_max_length,
                                                    early_stopping=_early_stopping)
                    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                    st.write('Summary')
                    st.success(output[0])


            if st.button('Submit Text'):
                run_model(text)

            title_SOr = """
            <div style="background-color:#eebd8a;padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">OR \nClick below to use a data frame that contains news stories on the Shell Corporation from various news sources.</h3>
            """
            st.markdown(title_SOr, unsafe_allow_html=True)
            if st.button("Summarize Dataframe"):
                st.write("Loading \----------------------")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
                t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
                data = pd.read_csv('guardian_publications.csv')
                data = data.head(3)
                data.drop(['Unnamed: 0', 'authors', 'title', 'authors', 'keywords', 'publish_date'], axis = 1, inplace = True) 
                result = []
                for i in range(len(data["text"])):
                    result.append(str(data["text"][i]).replace('\n', ''))                    
                data["input_text"] = result
                result1 = []
                for i in range(len(data["input_text"])):
                    result1.append( ' '.join(data["input_text"][i].split()))                    
                data["input_text"] = result1                    
                result2 = []
                for i in range(len(data["input_text"])):
                    result2.append(t5_tokenizer.encode(data["input_text"][i], return_tensors="pt").to(device))                    
                data["input_tokenized"] = result2                    
                summary_task = torch.tensor([[21603, 10]]).to(device)
                result3 = []
                for i in range(len(data["input_tokenized"])):
                    result3.append(torch.cat([summary_task, data["input_tokenized"][i]], dim=-1).to(device))                    
                data["input_tokenized"] = result3
                result4 = []
                for i in range(len(data["input_tokenized"])):
                    result4.append(t5_model.generate(data["input_tokenized"][i],
                                                num_beams=_num_beams,
                                                no_repeat_ngram_size=_no_repeat_ngram_size,
                                                length_penalty=_length_penalty,
                                                min_length=_min_length,
                                                max_length=_max_length,
                                                early_stopping=_early_stopping))                    
                data["summary_ids"] = result4                     
                result5 = []
                for i in range(len(data["summary_ids"])):
                    result5.append([t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                        data["summary_ids"][i]])                    
                data["Summrized Text"] = result5   
                st.write('Summary')
                st.write(data[["Summrized Text","text"]])

                df = pd.DataFrame(data[["Summrized Text","text"]])

                c29, c32, c31 = st.columns([1, 1, 2])

                with c29:

                    CSVButton = download_button(
                        df,
                        "Summarized_Dataframe.csv",
                        "Download to CSV",
                    )

        if choose == "-Q n A":
            title_SOqa = """
            <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">Question and Answer"""     
            st.markdown(title_SOqa, unsafe_allow_html=True) 

            predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

            # Create a text area to input the passage.
            passage = st.text_area("Passage", """Europe’s largest oil refinery suffered a malfunction, a potential source of jitters for the continent’s refined fuels market where supply has already been hit by industrial action.

            The compressor of fluid catalytic cracker unit 2 tripped on Oct. 12 due to the loss of power supply, according to a fire safety alert from the region’s Rjinmond Veilig service. Known as FCC units, the conversion plants are typically used to make refined products such as gasoline.

            Shell Plc’s Pernis plant near Rotterdam has been flaring elevated amounts of gas following the incident, triggering 200 complaints from the public, DCMR, an environmental regulator, said in a notice on its website. The plant is also an important source of diesel within Europe.

            The continent can ill afford material disruption to refined petroleum supply, given a European Union ban on purchases from Russia that’s due to start in early February. Strikes over pay in France have knocked out a swath of the nation’s fuelmaking, crunching supply.

            BP Plc is carrying out planned work on the FCC at its Rotterdam refinery, which is next to Pernis in Europe in terms of size.

            Read our blog on the European energy crisis

            Shell said in a statement that governments have been informed about the incident, but didn’t elaborate on what processing capacity was affected or what it would mean for fuel supply.

            “I can only tell you that we expect the nuisance will continue for the time being and that try to minimize the nuisance for the people in the vicinity,” a Shell spokesman said.

            — With assistance by April Roach, Rachel Graham and Jack Wittels""")

            # Create a text input to input the question.
            question = st.text_input("Question", "What technology was used?")

            # Use the predictor to find the answer.
            result = predictor.predict(question, passage)

            # From the result, we want "best_span", "question_tokens", and "passage_tokens"
            start, end = result["best_span"]
            question_tokens = result["question_tokens"]
            passage_tokens = result["passage_tokens"]

            # We want to render the paragraph with the answer highlighted.
            # We'll do that using `st.markdown`. In particular, for each token
            # if it's part of the answer span we'll **bold** it. Otherwise we'll
            # leave it as it.
            mds = [f"**{token}**" if start <= i <= end else token
                for i, token in enumerate(passage_tokens)]

            # And then we'll just concatenate them with spaces.
            if st.button("Find Answer"):
                st.write("Answer : "+result["best_span_str"])
                st.markdown(" ".join(mds))

            title_SOre = """
            <div style="background-color:#eebd8a;padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">OR \nClick below to use a data frame that contains news stories on the Shell Corporation from various news sources.</h3>
            """
            st.markdown(title_SOre, unsafe_allow_html=True)

            c29, c30, c31 = st.columns([1, 6, 1])

            with c30:

                uploaded_file = st.file_uploader(
                    "",
                    key="1",
                    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
                )

                if uploaded_file is not None:
                    file_container = st.expander("Check your uploaded .csv")
                    shows = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                    file_container.write(shows)

                else:
                    st.info(
                        f"""
                            👆 Upload a .csv file first. Sample to try: [ShellData.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/ShellData.csv)
                            """
                    )

                    st.stop()
            option = st.selectbox('Select Your questions:',('what is the cost and monetary value', 'what technology was used?', 'what resources used?'))
            st.write('You selected:', option)
            if st.button("Answer question"):

                def qesAns(df,questions = option):
                    df = df.head(5)
                    # """question and answer node meant to produce answers to artiles in a dataframe

                    # Args:
                    #     data: Data containing a text column.
                    # Returns:
                    #     data: a dataframe answering the asked question based on the articles in each row 
                    # """
                    def qestions(df,column,question):
                        result = []
                        for i in range(len(df[column])):
                            result.append(predictor.predict(passage=df[column][i], question=question)["best_span_str"])
                        return result

                    df[questions] = qestions(df,column= "text", question = questions)
                    return df[questions]
                res = qesAns(shows,questions= option)
                st.write(res)
                df = pd.DataFrame(res)

                c29, c33, c31 = st.columns([1, 1, 2])

                with c29:

                    CSVButton = download_button(
                        df,
                        "Summarized_Dataframe.csv",
                        "Download to CSV",
                    )
                st.stop()
 
        if choose == "-Entity recognition":
            title_SOer = """
            <div style="padding:10px;border-radius:10px;margin:10px; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">Locations and Orgnizations"""     
            st.markdown(title_SOer, unsafe_allow_html=True) 

            c29, c30, c31 = st.columns([1, 6, 1])

            with c30:

                uploaded_file = st.file_uploader(
                    "",
                    key="1",
                    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
                )

                if uploaded_file is not None:
                    file_container = st.expander("Check your uploaded .csv")
                    shows = pd.read_csv(uploaded_file)
                    shows = shows.drop_duplicates(subset=['text'], keep='last')
                    shows.dropna(subset=['text'], inplace=True)
                    shows.reset_index(inplace=True)
                    uploaded_file.seek(0)
                    file_container.write(shows)

                else:
                    st.info(
                        f"""
                            👆 Upload a .csv file first. Sample to try: [ShellData.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/ShellData.csv)
                            """
                    )


            if st.button("Find Locations and Orgnizations"):
                nlp_models = [
                    { 'name' : 'ner-elmo',
                        'url' : 'https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz'
                    },
                ]
                for nlp_model in nlp_models:
                    nlp_model['model'] = Predictor.from_path(nlp_model['url'])
                def locationOrganization(train):
                        train = train.head(5)
                        train = train.drop_duplicates(subset=['text'], keep='last')
                        def entity_recognition (sentence):
                            location = []
                            for nlp_model in nlp_models:
                                results =  nlp_model['model'].predict(sentence=sentence)
                                for word, tag in zip(results["words"], results["tags"]):
                                    if tag != 'U-LOC'and tag != 'B-LOC':
                                        continue
                                    else:
                                        # print([word])#(f"{word}")
                                        location.append(word)
                                # print()
                                return location

                        def entity_recognition_pe(sentence):
                            organisation = []
                            for nlp_model in nlp_models:
                                results =  nlp_model['model'].predict(sentence=sentence)
                                for word, tag in zip(results["words"], results["tags"]):
                                    if tag != 'U-ORG' and tag != 'B-ORG':
                                        continue
                                    else:
                                        # print([word])#(f"{word}")
                                        organisation.append(word)
                                # print()
                                return organisation
                        result = []
                        for i in range(len(train["text"])):
                            result.append(list(set(entity_recognition(train["text"][i]))))
                        re1 = []
                        for i in range(len(train["text"])):
                            re1.append(list(set(entity_recognition_pe(train["text"][i]))))
                        train["location"]=result
                        train["organisation"]=re1
                        train['location'] = [', '.join(map(str, l)) for l in train['location']]
                        train['organisation'] = [', '.join(map(str, l)) for l in train['organisation']]
                        return train[["text","location","organisation"]]
                res1 = locationOrganization(shows)
                st.write(res1)
                df = pd.DataFrame(res1)

                c87, c33, c31 = st.columns([1, 5, 2])

                with c87:

                    CSVButton = download_button(
                        df,
                        "Locations_and_Org_Dataframe.csv",
                        "Download to CSV",
                ) 
                op_ratings = st.radio("Map Data",("Locations","exit"))

                if op_ratings == "Locations":
                    def locvis(df):
                        a_list = []
                        a_list.extend(df['location'].tolist())
                        x = ()
                        for values in df.location.iteritems():
                                x += values
                        passage=str(x)
                        passage1 = passage.replace("'", "")
                        passage1 = ''.join([i for i in passage1 if not i.isdigit()])
                        def entity_recognition (sentence):
                            miscellaneous = []
                            person = []
                            organisation = []
                            loc = []
                            for nlp_model in nlp_models:
                                results =  nlp_model['model'].predict(sentence=sentence)
                                for word, tag in zip(results["words"], results["tags"]):
                                    if tag != 'U-LOC':
                                        continue
                                    else:
                                        loc.append(word)
                                return loc
                        coutries = entity_recognition (passage1)
                        from pandas import DataFrame
                        df = DataFrame (coutries,columns=['coutries'])
                        item_counts = df["coutries"].value_counts()
                        dfa = DataFrame (item_counts,columns=['coutries',"count"])
                        dfa.drop(['count'], axis = 1, inplace = True)
                        from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
                        def get_continent(col):
                            try:
                                cn_a2_code =  country_name_to_country_alpha2(col)
                            except:
                                cn_a2_code = 'Unknown' 
                            try:
                                cn_continent = country_alpha2_to_continent_code(cn_a2_code)
                            except:
                                cn_continent = 'Unknown' 
                            return (cn_a2_code, cn_continent)
                        dfa.reset_index(inplace=True)
                        dfa.rename(columns={'index': 'country','coutries': 'count'},
                            inplace=True, errors='raise')
                        import pycountry_convert as pc
                        country_continent_name=[]
                        for country in dfa['country']:
                            try:
                                country_alpha2 = pc.country_name_to_country_alpha2(country)
                                country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
                                name = pc.convert_continent_code_to_continent_name(country_continent_code)
                                country_continent_name.append(name)
                            except:
                                country_continent_name.append("Unknown")
                                
                        #     country_continent_name.append(name)

                        dfa['region'] = country_continent_name
                        # generate country code  based on country name 
                        import pycountry 
                        def alpha3code(column):
                            CODE=[]
                            for country in column:
                                try:
                                    code=pycountry.countries.get(name=country).alpha_3
                                # .alpha_3 means 3-letter country code 
                                # .alpha_2 means 2-letter country code
                                    CODE.append(code)
                                except:
                                    CODE.append('None')
                            return CODE
                        # create a column for code 
                        dfa['CODE']=alpha3code(dfa.country)
                        dfa.head()
                        import plotly.express as px

                        np.random.seed(12)
                        gapminder = dfa
                        gapminder['counts'] = np.random.uniform(low=100000, high=200000, size=len(gapminder)).tolist()

                        fig = px.choropleth(gapminder, locations="CODE",
                                            locationmode='ISO-3',
                                            color="count", 
                                            hover_name="CODE",
                                            color_continuous_scale=px.colors.sequential.Blues)

                        st.plotly_chart(fig, use_container_width=True)
                    st.write(locvis(df) )
                op_rating = st.radio("Bar chart",("Top 20 Organizations","exit."))

                if op_rating == "Top 20 Organizations":
                    #if rating_option == "Top 10 users by number of ratings":
                    st.image('resources/imgs/maps.png',use_column_width=True) 




                            
    if page_selection == "About us":
        with st.sidebar:
            choose = option_menu("About us:", ["-Developer Info","-Contact info"],
                                icons=['tree', 'kanban'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#f1f2f6"},
                "icon": {"color": "#eebd8a", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#585858"},		
                            }
                            )  
        if choose == "-Developer Info":
                title_about1 = """
                <div style="padding:10px;border-radius:10px;margin:10px;">
                <h1 style="color:black;text-align:center;"> - The Team -</h1>
                <h3 style="color:black;text-align:center;">We are a team of data science students from Explore Data Science Academy</h3>
                """

                contributors2 = """
                <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
                <h1 style="color:black;text-align:center;"> - Developers -</h1>
                """
                
                
                st.markdown(title_about1, unsafe_allow_html=True)
                st.markdown(contributors2, unsafe_allow_html=True)
                # st.image('resources/imgs/team members.png',use_column_width=True)
    if page_selection == "Contact us":
        with st.sidebar:
            choose = option_menu("Contact us", ["-Interactive online form"],
                                icons=['tree', 'kanban'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#f1f2f6"},
                "icon": {"color": "#eebd8a", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#585858"},		
                    }
                    ) 
        if choose == "-Interactive online form":
            title_SOio = """
            <iframe src="https://forms.gle/fQKDo4wNNqcw4dx49" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="1260px" width="700px" allowfullscreen></iframe>
            """     
            st.markdown(title_SOio, unsafe_allow_html=True)  




if __name__ == '__main__':
    main()
