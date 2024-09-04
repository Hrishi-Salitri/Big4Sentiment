# Reg stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


plt.style.use("ggplot")

# NLTK library for text analysis
import nltk

nltk.download("popular")

# For ROBERTA model from Huggingface, transfer learning for pretrained model on sensitivity
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import AutoTokenizer


# Read in data
path = "C:\\Users\hsali\OneDrive - University of Tennessee\Courses\BAS 476 - Python\glassdoor_reviews.csv"
df = pd.read_csv(path)


# -----------------------------Start---------------------------------------------


def Firm_Sentiment(firm):
    def pick_company_df(firm_name):
        firm_df = df[df["firm"] == firm_name]
        firm_df = firm_df.sample(5000)
        return firm_df

    test_df = pick_company_df(firm)

    # loading pretrained model from huggingface, transfer learning
    # using trained weights and apply to dataset
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def Roberta_PolarityScores(text):
        encoded_text = tokenizer(text, return_tensors="pt")
        modeled_text = model(**encoded_text)

        # Pull out data and normalize it to be able to standardize it
        scores = modeled_text[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            "Negative": scores[0],
            "Nuetral": scores[1],
            "Positive": scores[2],
        }

        return scores_dict

    def get_sentiment_df(df, ProCon="cons"):
        sentiment_scores = []
        takeout_indicies = []

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                text = row[ProCon]
                roberta_result = Roberta_PolarityScores(text)
                sentiment_scores.append(
                    {
                        "scores": roberta_result,
                        "ID": index,
                    }
                )
            except RuntimeError:
                print(f"broke for index value {index}, text was too big.")
                takeout_indicies.append(index)

        sentiment_df = pd.DataFrame(sentiment_scores)
        return sentiment_df, takeout_indicies

    # Get scores
    con_scores, ConsRemoveIndex = get_sentiment_df(test_df, "cons")
    pro_scores, ProsRemoveIndex = get_sentiment_df(test_df, "pros")

    # df1 MUST be the pros, df2 MUST be the cons
    def reshape_df(pro, con):
        def extract_values(row):
            return pd.Series(row["scores"])

        pro[
            [
                "pros_negative%",
                "pros_nuetral%",
                "pros_positive%",
            ]
        ] = pro.apply(extract_values, axis=1, result_type="expand")

        con[
            [
                "cons_negative%",
                "cons_nuetral%",
                "cons_positive%",
            ]
        ] = con.apply(extract_values, axis=1, result_type="expand")

        return pro, con

    def clean_df(pro, con):
        pro = pro.drop("scores", axis=1)
        con = con.drop("scores", axis=1)

        pro = pro[~pro["ID"].isin(ConsRemoveIndex)]
        con = con[~con["ID"].isin(ProsRemoveIndex)]

        return pro, con

    # Clean data for merge
    pro_scores, con_scores = reshape_df(pro_scores, con_scores)
    pro_scores, con_scores = clean_df(pro_scores, con_scores)

    both_scores = pro_scores.merge(con_scores, on="ID", how="left")

    test_df = test_df.reset_index()
    test_df = test_df.rename(columns={"index": "ID"})
    test_sentiment = both_scores.merge(test_df, on="ID", how="left")

    return test_sentiment


# -------------------------------------------------------------------------------------------------------------#

# Top 4 consulting firms data, took hours to run. Download 4 dataframes to use for App
# Original data over 20,000 for each company, subsetted to 5,000 rows, added a sentiment value for Pros and Cons comments


# Deloitte = Firm_Sentiment("Deloitte")
# Ey = Firm_Sentiment("EY")
# PwC = Firm_Sentiment("PwC")
# Kpmg = Firm_Sentiment("KPMG")

# Deloitte.to_csv("Deloitte_sentiment.csv", index=False)
# Ey.to_csv("EY_sentiment.csv", index=False)
# PwC.to_csv("PwC_sentiment.csv", index=False)
# Kpmg.to_csv("KPMG_sentiment.csv", index=False)
