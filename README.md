for Fundamentals of Machine learning @HU

## Dutch News Articles

### Datasets

I want to use the [Dutch News Articles](https://www.kaggle.com/maxscheijen/dutch-news-articles) dataset I found on Kaggle.com. This dataset contains all the articles published by the NOS as of the 1st of January 2010 till today. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**The dataset includes the following columns:**

- **datetime**: date and time of publication of the article.
- **title**: the title of the news article.
- **content**: the content of the news article.
- **category**: the category under which the NOS filed the article.
- **url**: link to the original article.

The dataset consists of **218860** unique rows.

By analyzing the title and content section, I'm able to generate more interesting datapoints. For example if the article has a positive or negative tone, or about who (politicians, public figures or political parties for example) they are writing.

I was also thinking that it could be interesting to combine it with the [Dutch Social media collection](https://www.kaggle.com/skylord/dutch-tweets). This dataset contains 10 files with around 271,342 tweets and was last updated on 4th of December this year. The tweets were hydrated using Twitter's API and then filtered for those which were in Dutch language and/or for users who had mentioned that they were from within Netherlands geographical borders.

### Y - What to predict

I have couple ideas on what to predict:

- has a positive or negative tone of voice correlations with the persons or subject in the published articles.
- predicting when a certain article about a certain topic (or about a certain person) is being published.
- predicting if someone is a stated as a bad or a good person in the published articles.

### Unit of Observation

I think this really depends on what I'm going to predict. I would like to have some feedback on this.

### X variables

What variables you think of using as a predictor: X variables. You should have at least 6 variables, preferably at least a dozen, and the more, the better. If you are using a text variable, a single text variable will suffice (more is possible). The texts have to be at least the length of a tweet and you should have several hundred of them.

I can at least use:

- time of publishing
- the title of the article
- the content of the article
- the category

I'm able to add more X variables by analyzing the title and the content of the articles. I still have to decide what to analyse.

### Difficulties

- I'm still not really sure about what to predict.
- And I'm new to machine learning and sometimes still feel a bit overwhelmed, sometimes it still feels a bit like magic. But I still think that with a bit practice and just diving into it, I can figure this out.
