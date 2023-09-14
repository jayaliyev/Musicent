# Musicent musical sentiment analysis application.
## Application is online in [here](https://musicent-63176680ee07.herokuapp.com/)
### App consists of two section:
* __Musical sentiment analysis for user given spotify artist.__<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plots a scatter plot of given artist's tracks as shown below.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;X axis represents valence value of tracks<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Y axis represents energy value of tracks<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data points in scatter plot colored by belonging album.<br/>
![](screenshots/artist_tracks_plot.png)<br/><br/>
* __Musical sentiment analysis for user's spotify playlists.__<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plots two bar plots. First one is for user's playlist tracks, Second one is for "Today's Hot Hits" playlist tracks <br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;X axis of bar plots represents mood of tracks.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Y axis is represenets value counts of corresponding moods.<br/><br/>
![](screenshots/user_tracks_barplots.png)<br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;And plots two scatter plots.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;First one is scatter plot of user's tracks.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; X axis is valence. Y axis is energy.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Explanation for second scatter plot is given in below screenshot (bottom right).<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; X and Y axis of this plot are determined by PCA algorithm.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Top three distinct tracks are determined by Isolation Forest outlier detection algorithm. <br/><br/>
![](screenshots/user_tracks_scatterplots.png)
