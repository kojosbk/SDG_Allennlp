
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

img = get_img_as_base64("image.png")

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
            st.markdown('Using BART and T5 transformer model')

            model = st.selectbox('Select the model', ('T5', 'BART'))

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

            col1, col2, col3 = st.columns(3)
            _num_beams = col1.number_input("num_beams", value=_num_beams)
            _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
            _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

            col1, col2, col3 = st.columns(3)
            _min_length = col1.number_input("min_length", value=_min_length)
            _max_length = col2.number_input("max_length", value=_max_length)
            _early_stopping = col3.number_input("early_stopping", value=_early_stopping)

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
            st.header("-Question & Anwser")

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
            st.header("Locations and Orgnizations") 
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

            # option = st.selectbox('Select Your questions:',('what is the cost and monetary value', 'what technology was used?', 'what resources used?'))
            # st.write('You selected:', option)
            if st.button("Find Locations and Orgnizations"):
                nlp_models = [
                    # { 'name' : 'ner-model',
                    # 'url': 'C:/Users/Silas_Dell/Downloads/Compressed/ner-elmo.2021-02-12.tar.gz'
                    # },
                    { 'name' : 'ner-elmo',
                        'url' : 'https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz'
                    },
                ]
                for nlp_model in nlp_models:
                    nlp_model['model'] = Predictor.from_path(nlp_model['url'])

                def locationOrganization(train):
                    train = train.head(3)
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

                with c33:

                    CSVButton = download_button(
                        df,
                        "Locations_and_Org_Dataframe.csv",
                        "Download to CSV",
                    ) 
                op_ratings = st.radio("Map Data",("Locations","exit"))

                if op_ratings == "Locations":
                    #if rating_option == "Top 10 users by number of ratings":
                    st.image('resources/imgs/map.png',use_column_width=True)  
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
        if choose == "Ratings":
            st.image('resources/imgs/EDA6.png',use_column_width=True)

            op_ratings = st.radio("Choose an option under ratings",("Top 20 users by number of ratings","Rating distribution","Relationship between the number of ratings a movie has and how highly it is rated"))

            if op_ratings == "Top 20 users by number of ratings":
                #if rating_option == "Top 10 users by number of ratings":
                st.image('resources/imgs/top_20rev.png',use_column_width=True)
                st.write('When we look at the top 20 users by the number of ratings, we can see that user 72315 is an outlier in that he or she has rated a disproportionately high number of films compared to the other users, with a difference of 9272 ratings between user 12952 and user 80974. As a result, our recommendation system "better knows" user 72315 and his or her preferences, making it simple for it to suggest films to them.')
            if op_ratings == "Relationship between the number of ratings a movie has and how highly it is rated":
                #if rating_option == "Top 10 users by number of ratings":
                st.image('resources/imgs/corr_rat_count.png',use_column_width=True)
                st.write('The scatter plot above indicates that the more ratings a film has, the more likely it is to obtain a high rating. This confirms our instinctive notion that films with better ratings tend to get more referrals from viewers. In other words, most people try to steer clear of negative comments.')
            
            # if op_ratings == "Top 10 users by number of ratings(No outlier)":
            # #if rating_option == "Top 10 users by number of ratings(No outlier)":
            #     st.image('resources/imgs/rating_no_outlier.png',use_column_width=True)
            #     st.write("Removing the outlier user 72315 we see that the rest of users have not rated an extreme number of movies comapred to each other.Now that we've looked into the number of ratings for each user, we can now investigate the distribution of ratings")
            #     st.write("Most review sites use a 1 to 5 star rating system, with")
            #     st.write("5 star : Excellent\n\n4.0 – 5.0 stars : Positive Reviews\n\n3.0 - 3.9 stars : Neutral Reviews\n\n1.0 - 2.9 star : Negative Reviews")
            if op_ratings == "Rating distribution":
            #if rating_option == "Rating distribution":
                st.image('resources/imgs/Ratings dist.png',use_column_width=True)
                st.write("When we look at the distribution of ratings, we can see that 4.0 is the most popular rating, making up 27% of all ratings, which suggests that most users have found most films to be good but not excellent—although no film can truly be excellent. The second most popular rating is 3.0, which indicates that many users have found the films they've seen to be neutral.")
                st.write("It's interesting to observe that the ratings are skewed to the left here, with more ratings on the right side of the bar graph. This might be due to the fact that individuals only prefer to review movies they enjoy, since they wouldn't bother to stay to the finish or score one they didn't like.")
                st.write("We can observe that the average movie rating is 3.5, indicating that the skewed distribution indicates that we have more neutral and favorable reviews.")

        if choose == "Movies":
           
            st.image('resources/imgs/predict bg.png',use_column_width=True)
            op_movies = st.radio("Choose an option under movies",("Top 20 most rated movies of all time","First Twenty Years with the Highest Numbers of Movies produced"))

            if op_movies == "Top 20 most rated movies of all time":
                st.image('resources/imgs/Top Twenty Rated Movies.png',use_column_width=True)
                st.write("Unsurprisingly, Shawshank Redemption, a 1994 American drama film written and directed by Frank Darabont, holds the record for highest box office gross. It is based on the 1982 Stephen King novel Rita Hayworth and Shawshank Redemption. Other timeless classics include The Matrix, which not only won 41 awards but also helped to define action filmmaking in the twenty-first century.\n\nHollywood's handling of action sequences was altered by The Matrix, which also helped to popularize the bullet time special effect. This method is used in the film's most famous moment, which leaves us in awe as Neo maneuvers his body to avoid an enemy's bullets while dodging their fire.\n\nIt's fascinating to observe that 21 of the top 25 movies of all time, or 84 percent of them, were published before the year 2000. Might this signify that people no longer rate movies, or could it just be because these films were produced so long ago that their rating counts have accumulated over time?\n\nFinding the highest-rated films of the twenty-first century is motivated by the discovery that 84% of the titles in our 20 most popular films were released before 2000.")
                test = '''<p float="left"><img src="https://www.themoviedb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg" width="200" height = 300/><img src="https://cdn.europosters.eu/image/750/posters/pulp-fiction-group-i1295.jpg" width="200" height = 300/>
                <img src="https://cps-static.rovicorp.com/2/Rights%20Managed/Belgacom/Forrest%20Gump/_derived_jpg_q90_310x470_m0/ForrestGump_EN.jpg" width="200" height = 300/></p>
                \n\n**Top three Most Rated Movies**\n- Shawshank Redemption (1994)<a href="https://www.youtube.com/watch?v=NmzuHjWmXOc&ab_channel=RottenTomatoesClassicTrailers"> Watch trailer</a>
                    \n- Pulp Fiction (1994)<a href="https://www.youtube.com/watch?v=s7EdQ4FqbhY&ab_channel=Movieclips"> Watch trailer</a>
                    \n- Forrest Gump (1994)<a href="https://www.youtube.com/watch?v=bLvqoHBptjg&ab_channel=ParamountMovies"> Watch trailer</a>'''
                st.markdown(test, unsafe_allow_html=True)
            if op_movies == "First Twenty Years with the Highest Numbers of Movies produced":
                st.image('resources/imgs/first_20years.png',use_column_width=True)
                st.write("The year 2015 had the largest number of films created, with over 1700. It was followed by the years 2016 and 2017, which both saw over 1500 films produced.")
                st.write("One thing to note is that a year like 2019 would have been expected to have a lot of movies released, but due to the outbreak of COVID 19, we observe a decline in movie releases in the year 2019.")

        if choose == "Directors":
        #if option_selection == "Directors":
            st.info("We start off with directors, A film director controls a film's artistic and dramatic aspects and visualizes the screenplay (or script) while guiding the technical crew and actors in the fulfilment of that vision. The director has a key role in choosing the cast members, production design and all the creative aspects of filmmaking\n\n\n\nEven though most people don't into finding our who director what movie to decide whether its going to be a good watch or not, there is a proportion of people that either watch the credits at the end of the movie or do research of each movie before they watch it, for these people director of a movie plays an import role in decided whether or not to watch a movie, for me personally I watch mroe series's than movies and but I know that if a series is directed by Chuck Lorre than I will definately love it.\n\nlet's start by finding our which directors have recieved the most number of ratings for their collective movies")
            
            op_director = st.radio("Choose an option under directors",("Top 3 most rated directors","Top 3 directors with most number of movies","10 highest rated director with over 10000 ratings","10 worst rated directors with over 10000 ratings"))

            if op_director == "Top 3 most rated directors":
                st.image('resources/imgs/top_25_most_D1.png',use_column_width=True)
                st.write("Topping the chart bar far we see Quentin Tarantino who has directed a total of 10 movies is an American film director, screenwriter, producer, and actor. His films are characterized by nonlinear storylines, aestheticization of violence, extended scenes of dialogue, ensemble casts, references to popular culture and a wide variety of other films, soundtracks primarily containing songs and score pieces from the 1960s to the 1980s, alternate history, and features of neo-noir film, One of Quentin Tarantino's highest rated movie Pulp fiction appreard in the top 10 best rated movies we saw ealier.\n\nwe also see familiar names like Stephen King who is also an author of horror, supernatural fiction, suspense, crime, science-fiction, and fantasy novels and directed a total of 23 movies among these movies is the movie we ponded a question of whether we can consider it as the best movie of all time, since it appeared top of the chart on both the Top 25 most rated movies of all time and Top 10 best rated movies of all time, Shawshank Redemption was based on Stephen King's novel.\n\n\n\nAfter seein the total number of ratings for each director its only natural that one wonders how many movies did each of these directors release, as this would contribute to the total number of ratings they have recieved, so lets find out which directors have released the most number of movies.")
            if op_director == "Top 3 directors with most number of movies":

                direct = '''### Top 3 Directors with most movies released

<img height = "238" width = 178 src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Luc_Besson_by_Gage_Skidmore.jpg/640px-Luc_Besson_by_Gage_Skidmore.jpg" alt="Photo of Luc Besson" class="GeneratedImage">  <img height = "238" width = 950 src="https://i.ibb.co/hL7J390/luc.jpg" alt="Movies of Tom Hanks" class="GeneratedImage">
</br>
<a href="https://en.wikipedia.org/wiki/Luc_Besson">Luc Paul Maurice Besson</a>  is a French filmmaker, writer, and producer of movies. The Big Blue, La Femme Nikita, and Subway were all movies he either directed or produced. </a>
</br>
</br>
<img height = "238" width = 178 src="https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQXYKDvhxIVt8R_yV3LLLZJ2LemcV860GqEgu9TKCDvGSDnHksM" alt="Photo of woody allen" class="GeneratedImage"> <img height = "238" width = 950 src="img\woody.JPG" alt="Movies of Tom Hanks" class="GeneratedImage">
</br>
<a href="https://en.wikipedia.org/wiki/Woody_Allen"> Woody Allen</a> is an American filmmaker, writer, actor, and comedian whose career spans more than six decades including several films that have won Academy Awards. </a>
</br>
</br>

<img height = "238" width = 178 src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3ie6FvZbpJx2VbSMzbGsFagq2wPgnNXJxSPQVTI6ofqhWv28AZKCUZIt54kEQHr9gfiI&usqp=CAU" alt="Photo of Stephen King" class="GeneratedImage"> <img height = "238" width = 950 src="img\king.JPG" alt="Movies of Tom Hanks" class="GeneratedImage"></br>
<a href="https://en.wikipedia.org/wiki/Stephen_King">Stephen King</a>  is an American writer of books in the genres of horror, science fiction, fantasy, suspense, and paranormal fiction. </a>
</br>
</br>


---

                '''
                st.markdown(direct, unsafe_allow_html=True)
                st.image('resources/imgs/Number of Movies Per director.png',use_column_width=True)
                st.write("Luc Besson and Woody Allen share the top rank with an equal amount of 26 films apiece. Stephen King is next. Having a total of 23 films, this time at number 2. We also notice some well-known names, including William Shakespeare, an English playwright, poet, and actor who is renowned as the greatest dramatist and writer in the English language. Additionally, Tyler Perry, a well-known producer, director, actor, screenwriter, dramatist, novelist, composer, businessman, and philanthropist, is most known for his Madea film series, which he not only directs but also stars in three roles.\n\n\n\nMost of the movies that were produced by the directors in the above bar plot have the genres Drama and Romance or a mixture of those two gernes popularly known as romantic comedies. Whether or not these two genres are the most succesful generes of highest rated genres is still to be investigated.")
            if op_director == "10 highest and worst rated director with over 10000 ratings":
            #if director_option == "10 highest and worst rated director with over 10000 ratings":
                st.image('resources/imgs/10_highest_rated_D3.png',use_column_width=True)
                st.image('resources/imgs/10_worst_directors_D4.png',use_column_width=True)
                st.write("Toping the chart of the best rated directors is Chuck Palahniuk, the director of Fight Club that recieved an average rating of 4.22 which had Action, Crime, Drama and thriller genres. The second spot is held by Christopher McQuarrie recieving an average rating of 4.19 for three movies he has directed namely Usual suspects, Way of the gun and Edge of Tomorrow with mix of genres Action, Crime and Thriller, this this shares some light on the question we posed earlier of whether people the most succesful genres were a mix of Drama, Romance or Comedy, as we see that our two best rated directors create blockbusters with mix of genres action and thriller. We will investigated these genres thoroughly at a later stage.\n\n\Looking at the worst rated directed we see that the lowest rated director is Jack Bernstein with an average rating of 2.84\n\n\n\nWe now move to the next factor that influences the perfrance of of viewers that is the genre of the movie.\n\n")
        
        
        if choose == "Genres":
        #if option_selection == "Genres":
            op_genre = st.radio("Choose an option under Genres",("Treemap of movie genres","Genre average rating over the years","Word cloud of movie genres"))
            #options_genres = ["Treemap of movie genres","Genre average rating over the years","Word cloud of movie genres"]
            #genre_options = st.selectbox('Choose option', options_genres)
            if op_genre == "Treemap of movie genres":
            #if genre_options == "Treemap of movie genres":
                st.image('resources/imgs/Treemap_G1.png',use_column_width=True)
                st.write("The genre treemap shows that Drama is the most popular genre with a genre count of 25606 followed by comedy with a count of 16870 as we initially suspected, We also see that Thriller and Romance follow suit. IMAX is by far the least popular genres with a count of 195 with Film-Noir following with a count of 353.\n\n\n\nWe have now seen the the most popular and least popular genres, lets now dig a little deeper into the genres and find out if whether the genre preference has changed throughout the years, to investigate this let's created an animated bar plot.")
            if op_genre == "Genre average rating over the years":
            #if genre_options == "Genre average rating over the years":
                st.video('resources/imgs/download.mp4')
                st.write("Right off the bat of the bet, the bar charr race shows us that there has been a change in genre preferences over the years")
                st.write("Stangely Animation was the best rated genre in 1995.\n\n\n\nIn 1996 Animation dropped to the 8th position and the Documentary became the most rated genre\n\n\n\n1997 Animation toped the char again and the following year Documentaty took over, seems between those 4 years the most prefered genres where Animation and Documentary, Strange times indeed...\n\n\n\nIn 1999 Crime movies started being popular and became the highest rated genre that year\n\n\n\nDrame took over the top spot in the year 2000\n\n\n\n2001 We see Fantasy, Crime and Drama taking the 1st. 2nd and 3rd spots respectively and we see these genres taking turns over the next couple of years until 2013 when Romance takes the lead and Documentaries become more popular and toping the chart in 2015.")
            if op_genre == "Word cloud of movie genres":
            #if genre_options == "Word cloud of movie genres":
                st.image('resources/imgs/Wordcloud_G3.png',use_column_width=True)

        if choose == "Title Cast":
           
        #if option_selection == "Title Cast":
            st.image('resources/imgs/Number of Movies Per Actor For Top 10 Actors.png',use_column_width=True)
            act = '''### Top 3 Movie Actors with Highest Number of movies released

<img height = "238" width = 178 src="https://cdn.britannica.com/77/191077-050-63262B99/Samuel-L-Jackson.jpg" alt="Photo of Tom Hanks" class="GeneratedImage">
<img height = "238" width = 950 src="https://i.ibb.co/x3452bJ/sam.jpg" alt="Movies of Tom Hanks" class="GeneratedImage"></br>
<a href="https://en.wikipedia.org/wiki/Tom_Hanks">Samuel Leroy Jackson </a>  is an American actor. One of the most widely recognized actors of his generation, the films in which he has appeared have collectively grossed over $27 billion worldwide, making him the highest-grossing actor of all time. </a>
</br>
</br>
<img height = "238" width = 178 src="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSItxlkc_a8e3O3T59cqB6Uw5iPRY5bJlmr8ZUt0KRPLObKcdTd" alt="Photo of Edward Norton" class="GeneratedImage">
<img height = "238" width = 950 src="https://i.ibb.co/qM46cdK/wilss.jpg" alt="Movies of Bruce Willis" class="GeneratedImage"></br>
<a href="https://en.wikipedia.org/wiki/Bruce_Willis">Bruce Willis</a>  is a famous American actor. In the 1970s, his acting career got its start on an off-Broadway theater. He rose to stardom in a starring role on the comedy-drama series Moonlighting (1985–1989), and he went on to feature in more movies, being known as an action hero for his performances as John McClane in the Die Hard trilogy (1988–2013) and other projects.</a>
</br>
</br>

<img height = "238" width = 178 src="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRpa6S_5nG_DBliLGNbMMqx_tSAxqmQqDbK26PsKIdZUPxZT017" alt="Photo of Steve Buscemi" class="GeneratedImage">
<img height = "238" width = 950 src="https://i.ibb.co/K7JbtwT/ste.jpg" alt="Movies of Leonardo DiCaprio" class="GeneratedImage"></br>
<a href="https://en.wikipedia.org/wiki/Steve_Buscemi">Steve Buscemi</a> is an American actor and film producer. Known for his work as a leading man in biopics and period films, he is the recipient of numerous accolades, including an Academy Award, a British Academy Film Award, and three Golden Globe Awards.  </a>
</br>
</br>
'''
            st.write("The likes of Samuel L. Jackson ,steve Buscemi ans Keith David are the most popular cast members according to the graph above.")
            st.markdown(act, unsafe_allow_html=True)

    #About
    if page_selection == "About":
        title_about = """
	    <div style="background-color:#eebd8a;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;"> - The Team -</h1>
        <h3 style="color:black;text-align:right;">We are a team of data science students from Explore Data Science Academy. This is our project for unsupervised sprint.</h3>
        """
        mission = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;"> - Our Mission - </h1>
        <h3 style="color:black;text-align:center;">To keep you entertained by helping you find movies you're most likely to enjoy&#128515</h3>
        """

        contributors = """
        <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;"> - Contributors -</h1>
        """
        
        
        st.markdown(title_about, unsafe_allow_html=True)
        st.markdown(mission, unsafe_allow_html=True)
        st.markdown(contributors, unsafe_allow_html=True)
        st.image('resources/imgs/team members.png',use_column_width=True)

    if page_selection == "Business Pitch":
        st.image('resources/imgs/BV_1.jpg',use_column_width=True)
        st.write("Some of the biggest companies in the world invested in streaming entertainment in the 21st century. The investment in streaming entertainment gave us platforms such as Netflix, Apple TV,, Disney Plus, Amazon prime and many more. These platforms are racking up millions of subscribers as the entire world is now streaming more than ever.")
        st.write("You may be wondering why these streaming platforms are attracting millions of subscribers, there are several reasons why people are leaning more towards streaming platforms. Streaming platforms have a lot of diverse content that can be consumed anywhere, anytime, and the subscribers are in total control of the rate at which they consume the content.")
        st.image('resources/imgs/BV_2.jpg',use_column_width=True)
        st.write("Another thing that is a major contributor in the rise and success of streaming platforms is their ability to recommend content that their users are most likely to watch and enjoy. They achieve this through the use of recommender algorithms. These algorithms ensure that each user is exposed to what they like.")
        st.image('resources/imgs/increasing.jpg',use_column_width=True)
        st.write("When doing exploratory data analysis we saw that the number of movies released increases exponentially each year. The exponential increase in the number of movies released means that streaming platforms need an excellent recommender algorithm to ensure that the movies reach the right audience.")
        st.image('resources/imgs/BV_L.jpg',use_column_width=True)
        st.write("This is where our recommender algorithm comes in. Our recommender algorithm will help with user retention by making tailored recommendations for each user. The user retention will ultimately result in a growth of the platform.")




if __name__ == '__main__':
    main()
