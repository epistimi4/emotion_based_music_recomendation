# Context-aware Music Recommendations #

Context and emotion-aware music recommendation sw

1. Compile a dataset of personal tracking metadata and self assessment mood report (e.g. <timestamp>, <context>, <activity>, <emotional_state> etc)
2. Parse songs from your personal Spotify account and align them to the data in 1.
3.Get metadata (emotional, danceability etc) from the spotify API for the songs in the database)
4.Get the raw audio data (if available in youtube). At this step the dataset is completed and contains: personal metadata, spotify playlists, spotify metadata, raw audio info.
5. Extract audio fratures from audio data
6. Measure correlations between personal tracking metadata and song metadata
7. Measure correlations between personal tracking metadata and raw audio features
8. (opt) 6 and 7 For different users

### References ###
* Baltrunas L. et al. InCarMusic: Context-Aware Music Recommendations in a Car. In: Huemer C., Setzer T. (eds) E-Commerce and Web Technologies. EC-Web 2011. Lecture Notes in Business Information Processing, vol 85. Springer, Berlin, Heidelberg. 2011
* C.-H. Yeh, H.-H. Lin, and H. Chang, An Efficient Emotion Detection Scheme for Popular Music. 2009.
* S. Pouyanfar and H. Sameti, “Music emotion recognition using two level classification,” in 2014 Iranian Conference on Intelligent Systems, ICIS 2014, 2014, pp. 1–6.
