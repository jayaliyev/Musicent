import sys, os, spotipy, dash, flask, json
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import spotipy.util as util
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from dash.dependencies import Input, Output, State
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


with open('app_token.txt', 'r') as file:
    tokens = file.read()

token_dict = json.loads(tokens)

client_credentials_manager = \
            SpotifyClientCredentials(client_id=token_dict['client_id'],
                                     client_secret=token_dict['client_secret'])
# Spotify API object
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    
def get_artist(name):
    """
    ----------
    name: 'Name of Spotify artist'
    ----------

    Returns: dictionary object.
        Given Spotify Artist's information.
    """
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']

    if len(items) > 0:
        return items[0]
    else: # If this artist does't exsists.
        return None

def get_artist_albums(artist):
    """
    ----------
    artist: artist's information that obtained by get_artist function.
    ----------
    Returns: dictionary object.
        Album names as keys, album ids as values.
    """
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    # By default artist_albums function can obtain up to 50 item in one try.
    # If there is more than 50 items, this function will return 'next'.
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])

    seen = set() # to avoid duplicates.
    album_dct = dict() # to keep name and id of album.
    albums.sort(key=lambda album:album['name'].lower())
    for album in albums:
        name = album['name']
        album_id = album['id']
        if name not in seen:
            seen.add(name)
            album_dct[name] = album_id
    return album_dct

def get_album_tracks(album_id):
    """
    Returns: dictionary object.
        Track ids as keys, track names as values.
    """
    tracks = []
    results = sp.album_tracks(album_id)
    tracks.extend(results['items'])
    # By default album_tracks function can obtain up to 50 item from spotify API in one try.
    # If there is more than 50 items, this function will return 'next'.
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    seen = set() # to avoid duplicates
    track_dct = dict() # to keep name and id of albums.
    tracks.sort(key=lambda track:track['name'].lower()) # sort tracks by name.
    for track in tracks:
        name = track['name']
        track_id = track['id']
        if name not in seen:
            seen.add(name)
            track_dct[track_id] = name
    return track_dct

def find_user_playlists(user_name):
    """
    Returns dictionary object.
        playlist names as keys, playlist ids as values.
    """
    playlists = sp.user_playlists(user_name)
    playlists_dict = dict() #contains playlist names as keys, playlist ids as values.
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            playlists_dict[playlist['name']] = playlist['uri'].split(':')[-1] # playlist id.
        if playlists['next']: # if there are items more than currently abtained items.
            playlists = sp.next(playlists)
        else:
            playlists = None # breaks wile loop.
        return playlists_dict

def show_user_tracks(results):
    """
    ------------
    results: dictionary object.
        results that obtained by spotipy.user_playlist_tracks function.
    ------------
    Returns: tuple of two dictionary objects; (artist_dict, track_dict).

    """
    # track_dict ==> {track_id: track_name}
    # artist_dict ==> {artist_id: artist_name}
    artist_dict, track_dict = dict(), dict()
    for i, item in enumerate(results['items']):
        track = item['track']
        artist_name = track['artists'][0]['name']
        artist_id = track['artists'][0]['id']
        track_name = track['name']
        track_id = track['id']
        artist_dict[track_id] = artist_name
        track_dict[track_id] = track_name

    return artist_dict, track_dict

def duplicate_dropper(df, field):
    """
    ------------
    Parametres:
    df: Pandas DataFrame object.
        dataset.
    field: string.
        column.
    ------------
    Takes a pandas DataFrame object and drops duplicates in choosen column.
    """
    if len(df[field]) != len(df[field].unique()):
        print('total rows: {}; unique rows: {}'.format(len(df[field]),
                                                       len(df[field].unique())))
        print('duplicates eleminated.')

        indicies_to_loc = []
        for index, value in zip(df[field].duplicated().index,
                                df[field].duplicated()):
            if str(value) == 'False':
                indicies_to_loc.append(index)

        new_df = df.loc[indicies_to_loc]
        return new_df
    else:
        print('total rows: {}; unique rows: {}'.format(len(df[field]),
                                                       len(df[field].unique())))
        print("this dataset doesn't include duplicates.")
        return df

def mood_detector(valence, energy):
    if valence > 0.50:
        if energy > 0.50:
            return 'Happy'
        else:
            return 'Relaxed'
    else:
        if energy > 0.50:
            return 'Angry'
        else:
            return 'Sad'

def todays_hot_hits():
    """
    Returns: Pandas DataFrame Object.
        dataframe that contains "Today's Hot Hits" playlist's tracks
        and their audio features.
    """
    def show_thh_tracks(results):
        """
        ------------
        results: dictionary object.
            results that obtained by spotipy.user_playlist_tracks function.
        ------------
        Returns: tuple of two dictionary objects; (artist_dict, track_dict).

        """
        # track_dict ==> {track_id: track_name}
        # artist_dict ==> {artist_id: artist_name}
        artist_dict, track_dict = dict(), dict()
        for i, item in enumerate(results['tracks']['items']):
            track = item['track']
            artist_name = track['artists'][0]['name']
            artist_id = track['artists'][0]['id']
            track_name = track['name']
            track_id = track['id']
            artist_dict[track_id] = artist_name
            track_dict[track_id] = track_name
        return artist_dict, track_dict

    track_dict_list = [] # list of dictionaries. Inner dictionaries ==> {track_id: track_name}
    artist_dict_list = [] # List of dictionaries. Inner dicts ==> {artist_id: artist_name}

    # playlist_id is "Today's Hot Hits" playlist's id.
    # Since "Today's Hot Hits" playlist is created by Spotify, user = 'spotify'.
    playlist_tracks = sp.user_playlist_tracks(user='spotify',
                playlist_id='37i9dQZF1DXcBWIGoYBM5M?si=yKc0b4axR8mcaoLj3rVyMQ')

    artist_dict, track_dict = show_thh_tracks(playlist_tracks)
    track_dict_list.append(track_dict)
    artist_dict_list.append(artist_dict)
    while playlist_tracks['tracks']['next']: # while there are more track information than currently obtained.
        playlist_tracks = sp.next(playlist_tracks)
        artist_dict, track_dict = show_thh_tracks(playlist_tracks)
        track_dict_list.append(track_dict)
        artist_dict_list.append(artist_dict)

    audio_feats = [] # list of dicts. Inner dicts contain audio feature's names as keys, and this feature's values as values.
    for i in range(len(track_dict_list)):
        trck_ids = list(track_dict_list[i].keys())
        audio_feats.append(sp.audio_features(trck_ids))

    df_list = [] # list of data frames. columns: names of audio features. rows: values of audia features.
    index = 0
    for i in range(len(audio_feats)):
        if audio_feats[i][0]: # if inner dictionary is not None.

            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))
    df = pd.concat(df_list)

    titles = []
    ids = []
    for i in range(len(track_dict_list)):
        for a in track_dict_list[i].keys():
            if a in list(df['id'].values): # if track id in values of "id" column.
                trck_names = track_dict_list[i][a]
                titles.append(trck_names)
                ids.append(a)
    track_titles = pd.DataFrame({'name':titles, 'id':ids})

    names = []
    ids = []
    for i in range(len(artist_dict_list)):
        for a in artist_dict_list[i].keys():
            if a in list(df['id'].values):
                names.append(artist_dict_list[i][a])
                ids.append(a)
    artist_names = pd.DataFrame({'artist_name':names, 'song_id':ids})

    first_df = df.merge(track_titles, left_on='id', right_on='id')
    first_df.drop(['type', 'key', 'uri', 'track_href',
                   'analysis_url', 'time_signature', 'mode', 'duration_ms'],
                    1, inplace=True)
    todays_hits_df =first_df.merge(artist_names,
                                    left_on='id',
                                    right_on='song_id')

    return todays_hits_df

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                server = server)


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.suppress_callback_exceptions = True
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.title = 'Moodify'

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.H1('Moodify App', style = {
                                    'color':'black',
                                    'backgroundColor':'white',
                                    'text-align':'center'
                        }
                    ), # Header of web site.
    html.Br(),
    dcc.Link('Sentiment Analysis For Your Favorite Artist',
             href='/page-1', style = {'font-size': '35px'}), # Link to plot_artist_songs function.

    html.Br(),
    dcc.Link('Sentiment Analysis For Your Playlists',
             href='/page-2', style = {'font-size': '35px'}), # link to plot_songs function.

    html.Div([html.Img(src = '/assets/background_image.jpg', height = '500',
                                    width = '37%', style={'margin-left':450})]),
    html.Div([html.H5('Developed by Badal Nabizade')],
                               style = {'color':'black','text-align':'center'}),
    html.Div([html.Div([html.H5(html.A('Contact',
                                       href='mailto:nabizadebadal@gmail.com'))],
                                        style = {'color':'black',
                                                 'text-align':'center',
                                                 'position': 'absolute'}),
        html.Div([html.H5(html.A('Source',
           href='https://github.com/badalnabizade/Moodify-Musical-Sentiment'))],
                                        style = {'color':'black',
                                                 'text-align':'right',
                                                 'position': 'absolute',
                                                 'left':'150px'})],

                                                 style={'width': '200px',
                                                        'heigt':'600px',
                                                        'position':'absolute',
                                                        'left':0, 'right':0,
                                                        'top':830, 'bottom':0,
                                                        'margin':'auto',
                                                        'max-width':'100%',
                                                        'max-height':'100%'})
    ], style = {'backgroundColor':'white', 'heigt':'100%', 'width':'100%'})

page_1_layout = html.Div([
    html.H1('Moodify'),
    # html.Div(children='''
    #     Symbol to graph:
    # '''),
    html.Div(dcc.Input(id='input',placeholder='Your favorite Spotify artist',
                        value='', type='text')),
    html.Button('Plot Tracks', id = 'button', className = 'row'),
    html.Div(id='output-graph'),
    html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Sentiment Analysis For Your Playlists', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
    [State(component_id = 'input', component_property = 'value')])

def plot_artist_songs(n_clicks, input_data):
    artist = get_artist(input_data)
    albums = get_artist_albums(artist)

    # list that contains track ids.
    tracks = []

    # list of tuples. first elemnt of inner tuples is list of track ids for particular album.
    # second element of inner tuples is name of that album.
    tracks_and_albums = []

    # list of dictionaries. inner dictionaries contain track id as keys, track name as values.
    track_dct_list = list()

    for i in albums.values():
        albums_reverse = {v:k for k,v in albums.items()}
        tracks.extend(get_album_tracks(i).keys())
        tracks_and_albums.append((list(get_album_tracks(i).keys()),
                                                            albums_reverse[i]))
        track_dct_list.append(get_album_tracks(i))

    # audio_features function can obtain data for up to 100 track.
    # if number of tracks is more than 100 it will return error.
    # So I will iterate through list of tracks, and will use maximum 100 tracks for one iteration.
    if len(tracks)>100:
        audio_feats = [] # list of dicts. Inner dicts contain audio feature's names as keys, and this feature's values as values.
        for i in range(0,len(tracks), 100):
            audio_feats.append(sp.audio_features(tracks[i:i+100]))

        df_list = [] # list of data frames. columns: names of audio features. rows: values of audia features.
        index = 0
        for i in range(len(audio_feats)):
            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))

    else: # In case, number of tracks <= 100.
        # dict that contains audio feature's names as keys, and this feature's values as values.
        audio_feats = sp.audio_features(tracks)
        df_list = [] # list of data frames. columns: names of audio features. rows: values of audia features.
        index = 0
        for i in range(len(audio_feats)):
            index+=1
            df_list.append(pd.DataFrame(audio_feats[i], index=[index]))

    df = pd.concat(df_list)

    def album_finder(track_id):
        for i in range(len(tracks_and_albums)):
            for t_id in tracks_and_albums[i][0]: # list of track ids in particular album.
                if t_id == track_id:
                    return tracks_and_albums[i][1] # name of that album.

    df['album'] = df['id'].apply(album_finder)

    names = []
    ids = []
    for i in range(len(track_dct_list)):
        for a in track_dct_list[i].keys():
            if a in list(df['id'].values):
                names.append(track_dct_list[i][a])
                ids.append(a)

    track_names = pd.DataFrame({'name':names, 'id':ids})

    final_df = df.merge(track_names, left_on='id', right_on='id')
    final_df.drop(['type', 'key', 'uri', 'track_href', 'analysis_url',
                   'time_signature', 'mode', 'duration_ms'], 1, inplace=True)


    trace0 = go.Scatter(
        x=[0.92, 0.05, 0.07, 0.92], # cordinates of below strings in x axis.
        y=[0.92, 0.05, 0.95, 0.05], # cordinates of below strings in y axis.
        text=['Happy', 'Sad', 'Angry', 'Relaxed'],
        mode='text',
        name = 'Categories',
        textfont = dict(
                        family='Old Standard TT, serif',
                        size=25,
                        color= 'rgba(0, 0, 0, 0.70)'
                                )
                            )

    trace1 = [go.Scatter(x=final_df[final_df['album'] == i]['valence'], # x axis repersents valence of particular track.
                y=final_df[final_df['album'] == i]['energy'], # y axis repersents energy of particular track.
                text=final_df[final_df['album'] == i]['name'], # names of data points in scatter plot are track names.
                mode='markers',
                opacity=0.7,
                marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
                },
                name=i # Legends of scatter plot consist of album names.
                ) for i in final_df.album.unique()]
    trace1.append(trace0)
    return html.Div([dcc.Graph(id = 'example',
              figure = {
                        'data':trace1,

                        'layout': {'clickmode': 'event+select',
                                   'height':650,
                                   'shapes': [
                                              {'type':'line', # line that dvide x axis two equal parts and paralel to y axis.
                                               'x0':0.5,
                                               'y0': 0,
                                               'x1':0.5,
                                               'y1':1},

                                              {'type':'line', # line that dvide y axis two equal parts and paralel to x axis.
                                               'x0': 0,
                                               'y0':0.5,
                                               'x1':1,
                                               'y1':0.5}
                                          ]
                                      }
                                  }
                              )
                          ], className='container', style={'maxWidth': '1000px'}
                      )

data_list = ["User Songs", "Today's Hot Hits"]
page_2_layout = html.Div([
    html.Div([
        html.H2('Moodify',
                style={'float': 'left'}),
        ]),
    dcc.Input(id='input',placeholder='Paste Your Spotify Profile Link...',
               value='', type='text'),
    html.Button('Plot Tracks', id = 'button', className = 'row',
                style = {'color':'rgba(186, 8, 8, 1)'}),
    html.H5("""To compare your tracks wtih "Today's Hot Hits" playlist, Select Today's Hot Hits also.""" ),
    dcc.Dropdown(id='dropdown-values',
                 options=[{'label': s, 'value': s}
                          for s in data_list],
                 value=['User Songs'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),

        html.Div([html.H5('Would you like to see top 3 distinct song ?'),
        html.Button(
        'Show Those', id = 'button-yes',
        value = 'yes', className = 'row',
        style = {
        'color':'rgba(186, 8, 8, 1)',
        'margin-left':115
        }
        ),],style={'width':'100%','margin-left':550,
                   'margin-right':10,'max-width':50000}
                 ),
        html.Br(),
        dcc.Link('Go back to home', href='/')
        ], className="container",style={'width':'98%','margin-left':10,
                                        'margin-right':10,'max-width':50000}
    )

@app.callback(
    Output(component_id='graphs', component_property='children'),
    [Input(component_id='button', component_property='n_clicks'),
    Input('dropdown-values', 'value'),
    Input('button-yes', 'n_clicks')],
    [State(component_id = 'input', component_property = 'value')])

def plot_songs(n_clicks, dropdown_value, button_yes, input_value):
    # function parameter input_value is user's spotify profile link.
    input_value = input_value.split('user/')[1].split('?')[0] # user's spotify id.
    playlists = find_user_playlists(input_value)

    track_dict_list = [] # list of dictionaries. Inner dictionaries ==> {track_id: track_name}
    artist_dict_list = [] # List of dictionaries. Inner dicts ==> {artist_id: artist_name}
    tracks_and_playlists = [] # List of tuples. first element of tuple is track id, second element is playlist name.
    for playlist in playlists.values(): # for playlist id in playlists ids.
        playlists_reverse = {v:k for k,v in playlists.items()}
        playlist_tracks = sp.user_playlist_tracks(user=input_value,
                                                    playlist_id=playlist)
        artist_dict, track_dict = show_user_tracks(playlist_tracks)

        track_dict_list.append(track_dict)
        artist_dict_list.append(artist_dict)
        tracks_and_playlists.append((list(track_dict.keys()),
                                        playlists_reverse[playlist]))

        while playlist_tracks['next']: # while there are more tracks than tracks obtained by user_playlist_tracks function.
            playlist_tracks = sp.next(playlist_tracks)
            artist_dict, track_dict = show_user_tracks(playlist_tracks)
            track_dict_list.append(track_dict)
            artist_dict_list.append(artist_dict)
            tracks_and_playlists.append((list(track_dict.keys()),
                                                playlists_reverse[playlist]))

    audio_feats = [] # list of dicts. Inner dicts contain audio feature's names as keys, and this feature's values as values.
    for i in range(len(track_dict_list)):
        audio_feats.append(sp.audio_features(list(track_dict_list[i].keys())))

    df_list = [] # list of data frames. columns: names of audio features. rows: values of audia features.
    index = 0
    for i in range(len(audio_feats)):
        if audio_feats[i][0]: # if inner dictionary is not None.

            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))

    df = pd.concat(df_list)

    def playlist_finder(track_id):
        """
        Takes: track id (string).
        Returns: corresponding playlist name (string).
        """
        for i in range(len(tracks_and_playlists)):
            for a in tracks_and_playlists[i][0]: # for track id in track ids
                if a == track_id:
                    return tracks_and_playlists[i][1] # return corresponding playlist name.

    df['playlist'] = df['id'].apply(playlist_finder)

    titles = []
    ids = []
    for i in range(len(track_dict_list)):
        for a in track_dict_list[i].keys(): # for id in track ids
            if a in list(df['id'].values): # if track id is in values of "id" column.
                titles.append(track_dict_list[i][a])
                ids.append(a)

    track_titles = pd.DataFrame({'name':titles, 'id':ids})

    names = []
    ids = []
    for i in range(len(artist_dict_list)):
        for a in artist_dict_list[i].keys():
            if a in list(df['id'].values):
                names.append(artist_dict_list[i][a])
                ids.append(a)

    artist_names = pd.DataFrame({'artist_name':names, 'song_id':ids})

    first_df = df.merge(track_titles, left_on='id', right_on='id')
    first_df.drop(['type', 'key', 'uri', 'track_href', 'analysis_url',
                    'time_signature', 'mode', 'duration_ms'], 1, inplace=True)
    user_songs_df =first_df.merge(artist_names, left_on='id', right_on='song_id')

    user_songs_df = duplicate_dropper(user_songs_df, 'id') # drop duplicate rows.

    user_songs_df.drop(['id'],1,inplace=True)

    user_songs_df['mood'] = list(map(mood_detector, user_songs_df['valence'],
                                                       user_songs_df['energy']))

    th_df = todays_hot_hits()
    th_df['mood'] = list(map(mood_detector, th_df['valence'], th_df['energy']))

    if len(dropdown_value) < 2: # if user don't want to see tracks of "Today's Hot Hits" playlist.
        class_choice = 'col s12' # fit one graph to width.
        data_dict = {dropdown_value[0]:user_songs_df['mood']}
    else:
        class_choice = 'col s12 m6 l6' # fit two graphs to width.
        data_dict = {dropdown_value[0]:user_songs_df['mood'],
                     dropdown_value[1]:th_df['mood']}

    graphs = []
    for val in dropdown_value: # dropdown values are "User songs" and "Today's Hot Hits"
        data = [go.Bar(
            x= data_dict[val].value_counts().index, # x axis represents moods (happy, sad, relaxed, angry).
            y=data_dict[val].value_counts().values, # y axis represents corresponding value counts of theese moods.
            marker=dict(
            color=[
                   'rgba(186, 8, 8, 1)',
                   'rgba(0,0,128,0.9)',
                   'rgba(246, 148, 30, 1)',
                   'rgba(0,255,0,1)'
                        ]
                      )
                   )]
        graphs.append(html.Div(dcc.Graph(
            id=val,
            animate=True,
            figure={'data': data,
                    'layout': go.Layout(title='{}'.format(val))
                    }
                ), className=class_choice
            )
        )

    trace0 = go.Scatter(
                        x=[0.9, 0.1, 0.1, 0.9], # cordinates of below strings in x axis.
                        y=[0.9, 0.1, 0.9, 0.1], # cordinates of below strings in y axis.
                        text=['Happy', 'Sad', 'Angry', 'Relaxed'],
                        mode='text',
                        name = 'Categories',
                        textfont = dict(
                                        family='Old Standard TT, serif',
                                        size=25,
                                        color='rgba(0, 0, 0, 0.68)'
                                                )
                                            )

    trace1 = [go.Scatter(
                x=user_songs_df[user_songs_df['playlist'] == i]['valence'], # x axis repersents valence of particular track.
                y=user_songs_df[user_songs_df['playlist'] == i]['energy'], # y axis repersents energy of particular track.
                text=user_songs_df[user_songs_df['playlist'] == i]['name'], # names of data points in scatter plot are track names.
                mode='markers',
                opacity=0.7,
                marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
                },
                name=i # Legends of scatter plot consist of playlist names.
                ) for i in user_songs_df.playlist.unique()]
    trace1.append(trace0)

    graphs.append(html.Div(dcc.Graph(
        id='scatter',
        animate=True,
        figure={'data': trace1,
                'layout': {'clickmode': 'event+select',
                           'shapes': [
                                      {'type':'line', # line that dvide x axis two equal parts and paralel to y axis.
                                       'x0':0.5,
                                       'y0': 0,
                                       'x1':0.5,
                                       'y1':1},

                                      {'type':'line', # line that dvide y axis two equal parts and paralel to x axis.
                                       'x0': 0,
                                       'y0':0.5,
                                       'x1':1,
                                       'y1':0.5}
                                    ]
                                }
                            }
                        ), className=class_choice
                    )
                )


    if button_yes: # if user clicks "Show Those" button.
        # PCA
        pca_df = user_songs_df.drop(['name','artist_name',
                                     'song_id', 'mood', 'playlist'], 1)
        pca_df = pca_df.dropna() # Data frame for PCA algorithm.
        pca_df_norm = (pca_df - pca_df.mean()) / pca_df.std() # normalize data frame.
        # Only keep two features that have largest variances (most important two features).
        pca = PCA(n_components=2, svd_solver='auto', iterated_power='auto')
        pca_res = pca.fit_transform(pca_df_norm.values)

        z1 = pca_res[:,0] # first feature (dimension)
        z2 = pca_res[:,1] # second feature (dimension)

        # Outlier Detection.
        # IsolationForest algorithm will be used with 1200 trees.
        forest = IsolationForest(n_estimators=1200, behaviour='new',
                                  contamination='auto')
        # preds will be 1 and -1. If given data point is outlier, it will be -1, 1 otherwise.
        preds = pd.Series(forest.fit_predict(pca_df))
        pca_df.reset_index(drop=False, inplace=True)
        outliers = preds[preds == -1].index

        outlier_data_points = []
        for i in range(len(outliers)):
            # 1d-array of values of outlier data point.
            outlier_track = pca_df.drop('index',1) \
                                  .loc[outliers[i]].values.reshape(1,-1).ravel()
            outlier_data_points.append(outlier_track)

        # list of average anomaly scores of outlier data points.
        anomaly_scores = forest.decision_function(np.array(outlier_data_points))
        outlier_dict = {v:i for i,v in enumerate(anomaly_scores)}

        indicies =  list() # indicies of top 3 outlier data points.
        for score in sorted(outlier_dict.keys())[:3]: # top 3 anomaly scores.
            outlier_data_idx = outliers[outlier_dict[score]]
            indicies.append(outlier_data_idx)

        outlier_track_names = user_songs_df \
                              .loc[pca_df.loc[indicies]['index'].values]['name']

        pca_df = pd.DataFrame({'z1':z1, 'z2':z2, 'name':user_songs_df['name']},
                               index=pca_df['index'])

        # General go.Scatter object, that contains both outlier and non-outlier data points.
        trace3 = [go.Scatter(
                            x = pca_df['z1'], # x axis represents first principal component.
                            y = pca_df['z2'], # y axis represents second principal component.
                            text = pca_df['name'], # names of data points in scatter plot are track names.
                            mode = 'markers',
                            name='tracks'
                            )]
        # go.Scatter object, that contains outlier data points.
        trace2 = go.Scatter(
                            x=pca_df.loc[outlier_track_names.index]['z1'], # first principal component of outlier.
                            y=pca_df.loc[outlier_track_names.index]['z2'], # second principal component of outlier.
                            text=pca_df.loc[outlier_track_names.index]['name'], # name of outlier track.
                            mode='markers',
                            marker={'size':10,
                                      'color':'rgba(255, 182, 193, .9)'},
                            name='Distinct Tracks'
                            )
        trace3.append(trace2)

        graphs.append((html.Div(dcc.Graph(
            id='new',
            animate=True,
            figure={'data': trace3,
                    'layout': go.Layout(clickmode= 'event+select',
                                        title="3 Most Different Tracks from User's General Music Taste.<br>(Note: This graph generated by PCA and Anomaly Detection Algorithm.<br>X and Y axis are two dimensons that provides highest variance about data.<br>Pink data points labaled as anomaly by anomaly detection algorithm.)",
                                        titlefont= dict(size = 14))}
            ), className=class_choice)))

    return graphs

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})
# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page


if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
