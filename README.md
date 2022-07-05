# movie-recommender-system

**Objective**:
a recommender system based on users' behaviors to recommend the content.

This project is based on [**Dataset**](https://github.com/lukkiddd-tdg/movielens-small) which applies techniques **collaborative and content based filtering**

## How it works and input & output

1. Path (Get): /recommendation, Query Parameter: ?user_id=user_id

***Example***
<img src="https://github.com/Suchawit/movie-recommender-system/blob/main/images/Screen%20Shot%202565-07-05%20at%2010.41.56.png" width="1000px"/>

2. Path (Get): /recommendation, Query Parameter: ?user_id=user_id&returnMetadata=true

***Example***
<img src="https://github.com/Suchawit/movie-recommender-system/blob/main/images/Screen%20Shot%202565-07-05%20at%2010.42.21.png" width="1000px"/>

3. Path (Get): /feature, Query Parameter: ?user_id=user_id

***Example***
<img src="https://github.com/Suchawit/movie-recommender-system/blob/main/images/Screen%20Shot%202565-07-05%20at%2010.50.25.png" width="1000px"/>

## How to improve in the future
collect date and time data that user watched the movie, and apply that data for probably 5 recent watched movies to be calculated and get recommendatation. Get IMDB score and how many people rated for each movie to be applied for improving recommenation system based on more population. Get top 10 of the watched movies in past week which might be viral, new and fun to watch
