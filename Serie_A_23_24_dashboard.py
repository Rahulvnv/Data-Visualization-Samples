import streamlit as st
import mplsoccer as mps
import pandas as pd
from PIL import Image
from urllib.request import urlopen
from mplsoccer import add_image
from ast import literal_eval
import numpy as np
from matplotlib.colors import to_rgba
from mplsoccer import Radar, FontManager
st.title("Serie A Dashboard 2023/24")
shots=['MissedShots','SavedShot','ShotOnPost']
goals=['Goal']
def_actions=['BallRecovery','Clearance','Tackle','Challenge','Interception','BlockedPass']
takeon=["TakeOn"]
touches=["BallTouch"]
@st.cache(suppress_st_warning=True,ttl=6*3600)
def eventloader(path):
    events=pd.read_pickle(path,compression='bz2')
    df = pd.read_csv("Serie A fixtures.csv")
    return events
df = pd.read_csv("Serie A fixtures.csv")
eventsdf=eventloader("test.bz2")
xT = pd.read_csv("xT_Grid.csv", header=None)
xT = np.array(xT)
xT_rows, xT_cols = xT.shape
def seconds_passed(min1,sec1, min2,sec2):
    if(min1==min2):
        return sec1-sec2
    elif(min1!=min2):
        x=(60*min1+sec1)-(60*min2+sec2)
        return x
@st.cache(suppress_st_warning=True,ttl=6*3600)
def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    """ Add carry events to whoscored events dataframe
    Function to read a whoscored-style events dataframe (single or multiple matches) and return an event dataframe
    that contains carry information.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        min_carry_length (float, optional): minimum distance required for event to qualify as carry. 5m by default.
        max_carry_length (float, optional): largest distance in which event can qualify as carry. 60m by default.
        min_carry_duration (float, optional): minimum duration required for event to quality as carry. 2s by default.
        max_carry_duration (float, optional): longest duration in which event can qualify as carry. 10s by default.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of events including carries
    """

    # Initialise output dataframe
    events_out = pd.DataFrame()

    # Carry conditions (convert from metres to opta)
    min_carry_length = 3.0
    max_carry_length = 60.0
    min_carry_duration = 1.0
    max_carry_duration = 10.0

    for match_id in events_df['matchId'].unique():

        match_events = events_df[events_df['matchId'] == match_id].reset_index()
        match_carries = pd.DataFrame()

        for idx, match_event in match_events.iterrows():

            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True

                while incorrect_next_evt:

                    next_evt = match_events.loc[next_evt_idx]

                    if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                        incorrect_next_evt = True

                    elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt[
                                'outcomeType'] == 'Unsuccessful')
                          or (next_evt['type'] == 'Foul')):
                        incorrect_next_evt = True

                    else:
                        incorrect_next_evt = False

                    next_evt_idx += 1

                # Apply some conditioning to determine whether carry criteria is satisfied

                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105*(match_event['endX'] - next_evt['x'])/100
                dy = 68*(match_event['endY'] - next_evt['y'])/100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt=seconds_passed(next_evt['minute'],next_evt['second'],match_event['minute'],match_event['second'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']

                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period

                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt

                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                                prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                        (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['playerName']=nex['playerName']
                    carry['home_team']=nex['home_team']
                    carry['away_team']=nex['away_team'] # I guess this comes as the error because in my events home & away events exist
                    carry['h_a']=nex['h_a']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(
                        ((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                         (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(
                        lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['matchId'] = nex['matchId']
                    carry['type'] = 'Carry'

                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

        # Rebuild events dataframe
        events_out = pd.concat([events_out, match_carries])

    return events_out
carr=insert_ball_carries(eventsdf)
carr['x1_bin'] = pd.cut(carr['x'], bins=xT_cols, labels=False)
carr['y1_bin'] = pd.cut(carr['y'], bins=xT_rows, labels=False)
carr['x2_bin'] = pd.cut(carr['endX'], bins=xT_cols, labels=False)
carr['y2_bin'] = pd.cut(carr['endY'], bins=xT_rows, labels=False)
carr['start_zone_value'] = carr[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
carr['end_zone_value'] = carr[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
carr['xT'] = carr['end_zone_value'] - carr['start_zone_value']
passes=eventsdf[eventsdf["type"]=="Pass"]
passes['x1_bin'] = pd.cut(passes['x'], bins=xT_cols, labels=False)
passes['y1_bin'] = pd.cut(passes['y'], bins=xT_rows, labels=False)
passes['x2_bin'] = pd.cut(passes['endX'], bins=xT_cols, labels=False)
passes['y2_bin'] = pd.cut(passes['endY'], bins=xT_rows, labels=False)
passes['start_zone_value'] = passes[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
passes['end_zone_value'] = passes[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
passes['xT'] = passes['end_zone_value'] - passes['start_zone_value']
passes=pd.concat([passes,carr])
options=["Match Report",'Player Report']
st.sidebar.header('Choose Viz Type- Player Report,Match Report')
selected_viz_type = st.sidebar.selectbox('Viz',options)
def count_special_character(string):
    # Declaring variable for special characters
    special_char = 0

    for i in range(0, len(string)):
        # len(string) function to count the
        # number of characters in given string.

        ch = string[i]

        # .isalpha() function checks whether character
        # is alphabet or not.
        if (string[i].isalpha()):
            continue

        # .isdigit() function checks whether character
        # is a number or not.
        elif (string[i].isdigit()):
            continue

        else:
            special_char += 1

    if special_char >= 1:
        return (special_char)
    else:
        print("There are no Special Characters in this String.")
def lineup_getter(events_df,k):
    test1=events_df[events_df["type"]=="FormationSet"]
    test2=test1[(test1["h_a"]==k)]
    x=pd.DataFrame(test2.iloc[0]["qualifiers"])
    list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
    list2=[]
    for i in range(len(list1)):
        plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[i])]["playerName"].unique().tolist()[0]
        list2.append(plyr)
    list3=[]
    for i in range(len(test2)):
        x=pd.DataFrame(test2.iloc[i]["qualifiers"])
        list1=[]
        list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
        list2=[]
        for j in range(len(list1)):
            plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[j])]["playerName"].unique().tolist()[0]
            list2.append(plyr)
        list3.append(list2)
    return list3[0]

def avg_dfgetter(events_df, k):
    test1 = events_df[events_df["type"] == "FormationSet"]
    test2 = test1[(test1["h_a"] ==k)]
    x = pd.DataFrame(test2.iloc[0]["qualifiers"])
    list1 = str.split(x[x["type"] == "InvolvedPlayers"]["value"].tolist()[0], ',', 11)[:11]
    list4 = str.split(x[x["type"] == "JerseyNumber"]["value"].tolist()[0], ',',
                      count_special_character(x[x["type"] == "JerseyNumber"]["value"].tolist()[0]))
    list5 = str.split(x[x["type"] == "InvolvedPlayers"]["value"].tolist()[0], ',',
                      count_special_character(x[x["type"] == "InvolvedPlayers"]["value"].tolist()[0]))
    df = pd.DataFrame(list4, list5)
    df = df.reset_index()
    df.columns = ["playerId", "Jersey_Num"]
    df["playerId"] = pd.to_numeric(df["playerId"])
    events_df["playerId"] = pd.to_numeric(events_df["playerId"])
    events_df = pd.merge(events_df, df, how="left", on=["playerId"])
    list2 = []
    for i in range(len(list1)):
        plyr = events_df[events_df["playerId"] == pd.to_numeric(list1[i])]["playerName"].unique().tolist()[0]
        list2.append(plyr)
    list3 = []
    for i in range(len(test2)):
        x = pd.DataFrame(test2.iloc[i]["qualifiers"])
        list1 = []
        list1 = str.split(x[x["type"] == "InvolvedPlayers"]["value"].tolist()[0], ',', 11)[:11]
        list2 = []
        for j in range(len(list1)):
            plyr = events_df[events_df["playerId"] == pd.to_numeric(list1[j])]["playerName"].unique().tolist()[0]
            list2.append(plyr)
        list3.append(list2)
    lineup = list3
    passes = events_df[events_df["type"] == "Pass"]
    passesh = passes[passes["h_a"] ==k]
    passesh = passesh[passesh["outcomeType"] == "Successful"]
    recep = []
    g = len(passesh) - 1
    for i in range(len(passesh)):
        if (i == g):
            recep.append(" ")
            continue
        recep.append(passesh.iloc[i + 1]["playerName"])
    passesh["pass_recepient"] = recep
    pass_number_raw = passesh[['minute', 'Jersey_Num', 'playerName', 'pass_recepient']]
    pass_number_raw['pair'] = pass_number_raw.playerName + pass_number_raw.pass_recepient
    pass_count = pass_number_raw.groupby(['pair']).count().reset_index()
    pass_count = pass_count[['pair', 'minute']]
    pass_count.columns = ['pair', 'number_pass']
    avg_loc_df = passesh[['teamId', 'Jersey_Num', 'playerName', 'x', 'y']]
    avgdf = avg_loc_df.groupby(['teamId', 'playerName']).agg({"x": "mean", "y": "mean"}).reset_index()
    return avgdf
def pass_network_dfgetter(events_df,k):
    test1=events_df[events_df["type"]=="FormationSet"]
    test2=test1[(test1["h_a"]==k)]
    x=pd.DataFrame(test2.iloc[0]["qualifiers"])
    list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
    list2=[]
    for i in range(len(list1)):
        plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[i])]["playerName"].unique().tolist()[0]
        list2.append(plyr)
    list3=[]
    for i in range(len(test2)):
        x=pd.DataFrame(test2.iloc[i]["qualifiers"])
        list1=[]
        list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
        list2=[]
        for j in range(len(list1)):
            plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[j])]["playerName"].unique().tolist()[0]
            list2.append(plyr)
        list3.append(list2)
    lineup=list3
    passes=events_df[events_df["type"]=="Pass"]
    passesh=passes[passes["h_a"]==k]
    passesh=passesh[passesh["outcomeType"]=="Successful"]
    recep=[]
    g=len(passesh)-1
    for i in range(len(passesh)):
        if (i==g):
            recep.append(" ")
            continue
        recep.append(passesh.iloc[i+1]["playerName"])
    passesh["pass_recepient"]=recep
    pass_number_raw = passesh[['minute', 'playerName', 'pass_recepient']]
    pass_number_raw['pair'] = pass_number_raw.playerName + pass_number_raw.pass_recepient
    pass_count = pass_number_raw.groupby(['pair']).count().reset_index()
    pass_count = pass_count[['pair', 'minute']]
    pass_count.columns = ['pair', 'number_pass']
    avg_loc_df = passesh[['teamId', 'playerName', 'x','y']]
    avgdf=avg_loc_df.groupby(['teamId','playerName']).agg({"x":"mean","y":"mean"}).reset_index()
    pass_merge = pass_number_raw.merge(pass_count, on='pair')
    pass_merge = pass_merge[['playerName', 'pass_recepient', 'number_pass']]
    pass_merge = pass_merge.drop_duplicates()
    pass_merge['width']=pass_merge['number_pass'] / pass_merge['number_pass'].max()
    passmerge=pd.merge(pass_merge,avgdf,how="left",on="playerName")
    a=len(passmerge)-1
    passmerge=passmerge.drop(passmerge.index[a])
    endx=[]
    endy=[]
    for i in range(len(passmerge)):
        endx.append(avgdf[avgdf["playerName"]==passmerge.iloc[i]["pass_recepient"]]["x"].values.tolist()[0])
        endy.append(avgdf[avgdf["playerName"]==passmerge.iloc[i]["pass_recepient"]]["y"].values.tolist()[0])
    passmerge["endx"]=endx
    passmerge["endy"]=endy
    lineup=lineup_getter(events_df,k)
    output1 = [x for x in passmerge["playerName"].unique().tolist() if not x in lineup or passmerge["playerName"].unique().tolist().remove(x)]
    for i in range(len(output1)):
        passmerge["playerName"]=passmerge["playerName"].replace(output1[i],np.nan)
        passmerge["pass_recepient"]=passmerge["pass_recepient"].replace(output1[i],np.nan)
    passmerge=passmerge.dropna(subset=["playerName"])
    passmerge=passmerge.dropna(subset=["pass_recepient"])
    min_transparency = 0.3
    color = np.array(to_rgba('cyan'))
    color = np.tile(color, (len(passmerge), 1))
    c_transparency = passmerge.number_pass / passmerge.number_pass.max()
    c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
    color[:, 3] = c_transparency
    passmerge['alpha'] = color.tolist()
    return passmerge


def match_report(events, shotdata, hlogo, alogo, games1):
    city_events = events
    games1 = games1
    carries = insert_ball_carries(city_events)
    avgdfh = avg_dfgetter(city_events, "h")
    avgdfa = avg_dfgetter(city_events, "a")
    pmergeh = pd.DataFrame()
    pmergeh = pass_network_dfgetter(city_events, "h")
    pmergea = pd.DataFrame()
    pmergea = pass_network_dfgetter(city_events, "a")
    lineuph = lineup_getter(city_events, "h")
    lineupa = lineup_getter(city_events, "a")
    passes = city_events[city_events["type"] == "Pass"]
    eventsh = passes[passes["h_a"] == "h"]
    eventsa = passes[passes["h_a"] == "a"]
    
    

    fm_rubik = FontManager(('https://github.com/google/fonts/blob/main/ofl/rubikmonoone/'
                            'RubikMonoOne-Regular.ttf?raw=true'))
    pmergehsize = pmergeh.groupby(["playerName"]).agg({"number_pass": 'sum'}).sort_values(by='number_pass',
                                                                                          ascending=False)
    pmergeasize = pmergea.groupby(["playerName"]).agg({"number_pass": 'sum'}).sort_values(by='number_pass',
                                                                                          ascending=False)
    pmergehsize = pmergehsize.reset_index()
    pmergeasize = pmergeasize.reset_index()
    pitch = mps.VerticalPitch(line_color="white", pitch_color="darkslategray", line_zorder=2, pitch_type='opta')
    pitch1 = mps.VerticalPitch(line_color="white", pitch_color="darkslategray", line_zorder=1, pitch_type='opta',
                               half=True, pad_bottom=15, positional=True, positional_color='white')
    pitch2 = mps.VerticalPitch(line_color="white", pitch_color="darkslategray", line_zorder=2, pitch_type='opta')
    pitch3 = mps.Pitch(line_color="white", pitch_color="darkslategray", line_zorder=1, pitch_type='opta')
    fig, axs = pitch.grid(nrows=1, ncols=1, title_height=0.1, axis=False, grid_width=0.9, figheight=17)
    fig1, axs1 = pitch1.grid(nrows=2, ncols=2, title_height=0.1, axis=False, grid_width=0.9, figheight=17)
    fig2, axs2 = pitch2.grid(nrows=1, ncols=1, title_height=0.1, axis=False, grid_width=0.9, figheight=17)
    fig3, axs3 = pitch3.grid(nrows=1, ncols=2, title_height=0.05, axis=False, grid_width=1, figheight=17)
    fig.set_facecolor("darkslategray")
    fig1.set_facecolor("darkslategray")
    fig2.set_facecolor("darkslategray")
    fig3.set_facecolor("darkslategray")
    ws = pmergeh["width"].values.tolist()
    ws = [i * 10 for i in ws]
    pitch.lines(pmergeh["x"], pmergeh["y"], pmergeh["endx"], pmergeh["endy"], lw=ws, color=pmergeh['alpha'], zorder=1,
                ax=axs["pitch"])
    ws1 = pmergea["width"].values.tolist()
    ws1 = [i * 10 for i in ws1]
    pitch.lines(pmergea["x"], pmergea["y"], pmergea["endx"], pmergea["endy"], lw=ws1, color=pmergea['alpha'], zorder=1,
                ax=axs2["pitch"])
    for i in range(len(lineuph)):
        pitch.scatter(avgdfh[avgdfh["playerName"] == lineuph[i]]["x"],
                      avgdfh[avgdfh["playerName"] == lineuph[i]]["y"],
                      s=pmergehsize[pmergehsize["playerName"] == lineuph[i]]['number_pass'] * 50, color="#00B2EE",
                      edgecolor='black', ax=axs["pitch"])
        try:
            pitch.annotate(lineuph[i].split(" ", 1)[1], xy=(
            avgdfh[avgdfh["playerName"] == lineuph[i]]["x"] - 2, avgdfh[avgdfh["playerName"] == lineuph[i]]["y"]),
                           c='white', va='center', ha='center', ax=axs["pitch"], fontsize=18, weight='bold')
        except:
            pitch.annotate(lineuph[i], xy=(
            avgdfh[avgdfh["playerName"] == lineuph[i]]["x"] - 2, avgdfh[avgdfh["playerName"] == lineuph[i]]["y"]),
                           c='white', va='center', ha='center', ax=axs["pitch"], fontsize=18, weight='bold')
    for i in range(len(lineupa)):
        pitch.scatter(avgdfa[avgdfa["playerName"] == lineupa[i]]["x"],
                      avgdfa[avgdfa["playerName"] == lineupa[i]]["y"],
                      s=pmergeasize[pmergeasize["playerName"] == lineupa[i]]['number_pass'] * 50, color="red",
                      edgecolor='black', ax=axs2["pitch"])
        try:
            pitch.annotate(lineupa[i].split(" ", 1)[1], xy=(
            avgdfa[avgdfa["playerName"] == lineupa[i]]["x"] - 3, avgdfa[avgdfa["playerName"] == lineupa[i]]["y"]),
                           c='white', va='center', ha='center', ax=axs2["pitch"], fontsize=18, weight='bold')
        except:
            pitch.annotate(lineupa[i], xy=(
            avgdfa[avgdfa["playerName"] == lineupa[i]]["x"] - 3, avgdfa[avgdfa["playerName"] == lineupa[i]]["y"]),
                           c='white', va='center', ha='center', ax=axs2["pitch"], fontsize=18, weight='bold')
    carriesa = carries[carries['h_a'] == 'a']
    carriesh = carries[carries['h_a'] == 'h']
    pitch1.lines(carriesh[carriesh["endX"] > 65]["x"], carriesh[carriesh["endX"] > 65]["y"],
                 carriesh[carriesh["endX"] > 65]["endX"], carriesh[carriesh["endX"] > 65]["endY"],
                 ax=axs1["pitch"][0][0], transparent=True, comet=True, color="cyan")
    pitch1.lines(carriesa[carriesa["endX"] > 65]["x"], carriesa[carriesa["endX"] > 65]["y"],
                 carriesa[carriesa["endX"] > 65]["endX"], carriesa[carriesa["endX"] > 65]["endY"],
                 ax=axs1["pitch"][0][1], transparent=True, comet=True, color="red")
    pitch1.scatter(carriesh[carriesh["endX"] > 65]["endX"], carriesh[carriesh["endX"] > 65]["endY"],
                   ax=axs1["pitch"][0][0], color="#00B2EE")
    pitch1.scatter(carriesa[carriesa["endX"] > 65]["endX"], carriesa[carriesa["endX"] > 65]["endY"],
                   ax=axs1["pitch"][0][1], color="red")
    shotdatah = shotdata[shotdata['h_a'] == 'h']
    shotdataa = shotdata[shotdata['h_a'] == 'a']
    pitch3.scatter((100 - shotdatah[shotdatah["type"].isin(shots)]["x"]),
                   100 - (shotdatah[shotdatah["type"].isin(shots)]["y"]),
                   s=500, alpha=0.7,
                   ax=axs3["pitch"][0], color="cyan", label=city_events["home_team"].unique()[0])
    pitch3.scatter(100 - (shotdatah[shotdatah["type"].isin(goals)]["x"] ),
                   100 - (shotdatah[shotdatah["type"].isin(goals)]["y"]),
                   s=500, ax=axs3["pitch"][0],
                   marker='football')

    pitch3.scatter(shotdataa[shotdataa["type"].isin(shots)]["x"] ,
                   shotdataa[shotdataa["type"].isin(shots)]["y"],
                   s=500, alpha=0.8,
                   ax=axs3["pitch"][0], color="#FF69B4", label=city_events["away_team"].unique()[0])
    pitch3.scatter(shotdataa[shotdataa["type"].isin(goals)]["x"],
                   shotdataa[shotdataa["type"].isin(goals)]["y"] ,
                   s=500, ax=axs3["pitch"][0],
                   marker='football')

    passes = city_events[(city_events["type"] == "Pass")]
    passes['x1_bin'] = pd.cut(passes['x'], bins=xT_cols, labels=False)
    passes['y1_bin'] = pd.cut(passes['y'], bins=xT_rows, labels=False)
    passes['x2_bin'] = pd.cut(passes['endX'], bins=xT_cols, labels=False)
    passes['y2_bin'] = pd.cut(passes['endY'], bins=xT_rows, labels=False)
    passes['start_zone_value'] = passes[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    passes['end_zone_value'] = passes[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    passes['xT'] = passes['end_zone_value'] - passes['start_zone_value']
    passesh = passes[passes['h_a'] == 'h']
    passesa = passes[passes['h_a'] == 'a']
    kp = passes[passes['passKey'] == True]
    pitch.arrows(kp[kp['h_a'] == 'h']['x'], kp[kp['h_a'] == 'h']['y'], kp[kp['h_a'] == 'h']['endX'],
                 kp[kp['h_a'] == 'h']['endY'], color='#00FF00', ax=axs1["pitch"][1][0])
    pitch.arrows(kp[kp['h_a'] == 'a']['x'], kp[kp['h_a'] == 'a']['y'], kp[kp['h_a'] == 'a']['endX'],
                 kp[kp['h_a'] == 'a']['endY'], color='#00FF00', ax=axs1["pitch"][1][1])

    title = axs1['title'].text(0.5, 1, '\n\n\n' + games1['score'].str.split(':', 2).tolist()[0][0] + ' - ' +
                               games1['score'].str.split(':', 2).tolist()[0][1],
                               ha='center', va='center', fontsize=40, color='white', weight="bold")
    add_image(hlogo, fig1, left=0.2, bottom=0.865, width=0.2, height=0.08)
    add_image(alogo, fig1, left=0.6, bottom=0.865, width=0.2, height=0.08)
    axs1["pitch"][0][0].set_title('Final 3rd Carries', color="white", fontsize=20, weight='bold')
    axs1["pitch"][0][1].set_title('Final 3rd Carries', color="white", fontsize=20, weight='bold')
    axs1["pitch"][1][0].set_title(city_events["home_team"].unique()[0] + ' ' + 'Key Passes', color="white", fontsize=20,
                                  weight='bold')

    axs1["pitch"][1][1].set_title(city_events["away_team"].unique()[0] + ' ' + 'Key Passes', color="white", fontsize=20,
                                  weight='bold')
    axs['title'].text(0.15, 0.1,
                      city_events["home_team"].unique()[0] + ' Pass Network ' + '\n(Scatter-size=Number of Passes)',
                      color="white", fontsize=22, weight='bold')
    axs2['title'].text(0.15, 0.1,
                       city_events["away_team"].unique()[0] + ' Pass Network ' + '\n(Scatter-size=Number of Passes)',
                       color="white", fontsize=22, weight='bold')

    eventsh = passes[passes["h_a"] == "h"]
    eventsa = passes[passes["h_a"] == "a"]
    fig5, ax5 = plt.subplots(figsize=(25, 17))
    ax5.set_facecolor('darkslategray')
    fig5.set_facecolor('darkslategray')
    t1 = passes[passes['h_a'] == 'h']['minute']
    t2 = passes[passes['h_a'] == 'a']['minute']
    mu1 = passes[passes['h_a'] == 'h'][['xT', 'minute']]
    mu2 = passes[passes['h_a'] == 'a'][['xT', 'minute']]
    mu1xt = []
    for i in t1.unique().tolist():
        mu1xt.append(mu1[mu1['minute'] == i]['xT'].mean())
    mu2xt = []
    for i in t2.unique().tolist():
        mu2xt.append(0 - mu2[mu2['minute'] == i]['xT'].mean())

    ax5.plot(t1.unique().tolist(), mu1xt, lw=2, label=city_events["home_team"].unique()[0] + ' xT Momentum',
             color='#FF1493')
    ax5.plot(t2.unique().tolist(), mu2xt, lw=2, label=city_events["away_team"].unique()[0] + ' xT Momentum',
             color='#00BFFF')
    ax5.fill_between(t1.unique().tolist(), 0, mu1xt, facecolor='#FF1493', alpha=0.4)
    ax5.fill_between(t2.unique().tolist(), mu2xt, 0, facecolor='#00BFFF', alpha=0.4)
    ax5.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=True, bottom=True, color='white')
    ax5.set_title('xT Momentum', color='white', weight='bold', fontsize=20)
    ax5.legend(loc='upper left')
    ax5.set_xlabel('Minute', color='white', weight='bold', fontsize=13)
    ax5.set_ylabel('xT')
    ax5.axvline(x=45, color='white')
    ax5.xaxis.label.set_color('white')
    ax5.tick_params(axis='x', colors='white')
    ax5.spines['left'].set_color('white')  # setting up Y-axis tick color to red
    ax5.spines['top'].set_color('white')
    ax5.spines['right'].set_color('white')  # setting up Y-axis tick color to red
    ax5.spines['bottom'].set_color('white')

    fig5.savefig('xT momentum.jpeg')
    xtm = Image.open('xT momentum.jpeg')
    add_image(xtm, fig3, left=0.43, bottom=0.125, width=0.67, height=0.8)
    axs3['pitch'][0].legend(loc='upper left', fontsize=25)
    title = axs1['title'].text(0.5, 1.1,
                               city_events["home_team"].unique()[0] + ' vs ' + city_events["away_team"].unique()[
                                   0] + ' Report 2023/24 (by @Rahulvn5)',
                               ha='center', va='center', fontsize=23, color='white', weight="bold")
    title = axs3['title'].text(0.5, 1.1, 'Shots Data and xT Momentum',
                               ha='center', va='center', fontsize=23, color='white', weight="bold")
    fig1.savefig("fig1mreport.jpeg")
    fig.savefig("figmreport.jpeg")
    fig2.savefig("fig2mreportgame.jpeg")
    fig3.savefig("fig3mreportgame.jpeg")
    import cv2
    img1 = cv2.imread('figmreport.jpeg')
    img2 = cv2.imread('fig1mreport.jpeg')
    img3 = cv2.hconcat([img1, img2])
    cv2.imwrite("fig4mreport.jpeg", img3)
    img1 = cv2.imread('fig4mreport.jpeg')
    img2 = cv2.imread('fig2mreportgame.jpeg')
    img3 = cv2.hconcat([img1, img2])
    cv2.imwrite("fig5mreport.jpeg", img3)
    img1 = cv2.imread('fig5mreport.jpeg')
    img2 = cv2.imread("fig3mreportgame.jpeg")
    img2 = cv2.resize(img2, (2377, 1224))
    img3 = cv2.vconcat([img1, img2])
    cv2.imwrite("fig6mreport.jpeg", img3)
    img8 = Image.open("fig6mreport.jpeg")
    return (img8)
if(selected_viz_type=="Player Report"):
    players = eventsdf["playerName"].unique().tolist()
    players = [x for x in players if str(x) != 'nan']
    st.sidebar.header('Player Input Tab')
    selected_player = st.sidebar.selectbox('Player', players)
    hal = eventsdf[eventsdf["playerName"] == selected_player]
    hal1 = passes[passes["playerName"] == selected_player]
    if(hal1.iloc[0]['h_a']=='h'):
        team=hal1.iloc[0]['home_team']
    else:
        team = hal1.iloc[0]['away_team']
    mids=events_df['matchId'].unique().tolist()
    mins=0
    for i in mids:
        eves=hal[hal['matchId']==i]
        mins=mins+eves['minute'].max()
    hal2=carr[carr['playerName']==selected_player]
    hal3=pd.concat([hal1,hal2])
    pitch = mps.VerticalPitch(line_color="white", pitch_color="black", line_zorder=2, pitch_type='opta')
    fig, axs = pitch.grid(nrows=2, ncols=3, title_height=0.1, axis=False, grid_width=1, figheight=30)
    pitch.lines(0, 100, 100, 100, ax=axs['pitch'][0][0], color='black')
    pitch.lines(0, 0, 100, 0, ax=axs['pitch'][0][0], color='black')
    pitch.lines(100, 100, 100, 0, ax=axs['pitch'][0][0], color='black')
    pitch.lines(50, 100, 50, 0, ax=axs['pitch'][0][0], color='black')
    pitch.lines(0, 100, 0, 0, ax=axs['pitch'][0][0], color='black')
    import matplotlib.patches as mlp
    axs['pitch'][0][0].add_artist(mlp.Circle((50, 50), 15, color='black', ec="none", zorder=2))
    axs['pitch'][0][0].add_artist(mlp.Rectangle((80, 80), -60, 20, color='black', ec="none", zorder=2))
    axs['pitch'][0][0].add_artist(mlp.Rectangle((80, 0), -60, 20, color='black', ec="none", zorder=2))
    pitch.lines(0, 100, 100, 100, ax=axs['pitch'][1][0], color='black')
    pitch.lines(0, 0, 100, 0, ax=axs['pitch'][1][0], color='black')
    pitch.lines(100, 100, 100, 0, ax=axs['pitch'][1][0], color='black')
    pitch.lines(50, 100, 50, 0, ax=axs['pitch'][1][0], color='black')
    pitch.lines(0, 100, 0, 0, ax=axs['pitch'][1][0], color='black')
    axs['pitch'][1][0].add_artist(mlp.Circle((50, 50), 15, color='black', ec="none", zorder=2))
    axs['pitch'][1][0].add_artist(mlp.Rectangle((80, 80), -60, 20, color='black', ec="none", zorder=2))
    axs['pitch'][1][0].add_artist(mlp.Rectangle((80, 0), -60, 20, color='black', ec="none", zorder=2))
    fig.set_facecolor("black")
    axs["pitch"][0][0].text(x=45, y=110, s='Player Information',
                            size=25, color='white',
                            va='center', ha='center', weight='bold')
    axs["pitch"][0][0].text(x=45, y=75, s='Name: ' + selected_player,
                            size=25, color='white',va='center', ha='center',weight='bold')
    axs["pitch"][0][0].text(x=45, y=65, s='Club:' + team,
                            size=25, color='black',
                            va='center', ha='center', weight='bold')
    axs["pitch"][0][0].text(x=45, y=55, s='Mins Played =' + str(mins),
                            size=25, color='black',
                            va='center', ha='center', weight='bold')
    import cmasher as cmr
    import matplotlib.patheffects as path_effects

    bin_statistic1 = pitch.bin_statistic_positional(hal3[hal3['xT'] > 0]['x'], hal3[hal3['xT'] > 0]['y'],
                                                    statistic='count',
                                                    positional='full', normalize=True)
    pitch.heatmap_positional(bin_statistic1, ax=axs['pitch'][1][2], cmap='Blues', edgecolors='black')
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    labels = pitch.label_heatmap(bin_statistic1, color='white', fontsize=18,
                                 ax=axs['pitch'][1][2], ha='center', va='center',
                                 str_format='{:.0%}', path_effects=path_eff)
    pitch.kdeplot(hal[hal['type'].isin(def_actions)]['x'], hal[hal['type'].isin(def_actions)]['y'], fill=True,
                  cmap=cmr.voltage_r, ax=axs['pitch'][1][1], levels=100)
    succ = 0
    unsucc = 0
    hal = hal[hal['passCorner'] == False]
    list1 = hal[hal['type'] == 'BallRecovery'].index.tolist()
    for i in list1:
        k = i + 1
        try:
            if ((hal[hal.index == k]['type'].unique().tolist()[0] == 'Pass') & (
                    hal[hal.index == k]['outcomeType'].unique().tolist()[0] == 'Successful')):
                pitch.arrows(hal[hal.index == k]['x'], hal[hal.index == k]['y'],
                             hal[hal.index == k]['endX'], hal[hal.index == k]['endY'], color='#7CFC00', width=3.4,
                             ax=axs['pitch'][1][1], label='Post Recovery Successful Pass')
                if (succ < 1):
                    axs['pitch'][1][1].legend(loc='upper left')
                succ += 1
            elif ((hal[hal.index == k]['type'].unique().tolist()[0] == 'Pass') & (
                    hal[hal.index == k]['outcomeType'].unique().tolist()[0] == 'Unsuccessful')):
                unsucc += 1

        except:
            continue
    axs["pitch"][1][1].set_title('Def. Actions Heatmap & Post Recovery Passes' + '\n' + 'Ball Retention rate % =' + str(
        round((succ / (succ + unsucc) * 100), 2)), color='black', weight='bold', size=20)
    shotdata = hal[hal["type"].isin(shots + goals)]
    pitch.scatter(shotdata[shotdata["type"].isin(shots)]["x"], shotdata[shotdata["type"].isin(shots)]["y"],
                  s=600, ax=axs["pitch"][0][2], color="red", edgecolor="black", label="Shot")
    pitch.scatter(shotdata[shotdata["type"].isin(goals)]["x"], shotdata[shotdata["type"].isin(goals)]["y"],
                  marker="football", c="white", edgecolor="black",
                  s=600, ax=axs["pitch"][0][2], label="Goal")
    txt = axs["pitch"][0][2].text(x=50, y=30
                                  , s=' Total Shots=' + '38' + '\n npxG Accumulated=0.16 Per90 \n Goals Scored=5',
                                  size=20, color='black',
                                  va='center', ha='center', weight='bold')
    pitch.arrows(hal[hal['assist'] == True]['x'], hal[hal['assist'] == True]['y'],
                 hal[hal['assist'] == True]['endX'], hal[hal['assist'] == True]['endY'], ax=axs['pitch'][0][1],
                 color='red', label='Assist', width=6)
    carr1=hal2
    pitch.lines(carr1[(carr1['endX'] - carr1['x']) > 10]['x'], carr1[(carr1['endX'] - carr1['x']) > 10]['y'],
                carr1[(carr1['endX'] - carr1['x']) > 10]['endX'],
                carr1[(carr1['endX'] - carr1['x']) > 10]['endY'], ax=axs['pitch'][0][1], comet=True, color='#BF3EFF',
                linestyle='--')
    pitch.scatter(carr1[(carr1['endX'] - carr1['x']) > 10]['endX'], carr1[(carr1['endX'] - carr1['x']) > 10]['endY'],
                  color='#BF3EFF', s=100, ax=axs['pitch'][0][1])
    axs["pitch"][1][2].set_title('Expected Threat Positional Heatmap', color='black', weight='bold', size=25)
    axs["pitch"][0][1].set_title('Progressive Carries & Assists', color='black', weight='bold', size=25)
    axs["pitch"][0][2].set_title('Shots & Goals', color='black', weight='bold', size=25)
    axs['pitch'][0][1].legend(loc='upper left')
    juv1 = Image.open("Logos/" + team + ".png")
    add_image(juv1, fig, left=0.08, bottom=0.7, width=0.22, height=0.16)
    axs['title'].text(0.14, 0.6, selected_player+ " Player Report for Season 2023/24 (made by:@Rahulvn5)", color="black",
                      weight='bold', fontsize=30)
    st.pyplot(fig, axs)
if(selected_viz_type=="Match Report"):
    matchDD = [{'label': row["home_team"] + " - " + row["away_team"],
                "value": row["matchId"]}
               for i, row in df.drop_duplicates(subset=["matchId"]).iterrows()]
    list1 = []
    for i in range(len(matchDD)):
        list1.append(matchDD[i]['label'])
    st.sidebar.header('Match Input Tab')
    selected_match = st.sidebar.selectbox('Match', list1)


    def matchId_giver(label):
        for i in range(len(matchDD)):
            if (matchDD[i]['label'] == label):
                a = matchDD[i]['value']
        return (a)
    team_plot_data = eventsdf[eventsdf["matchId"] == matchId_giver(selected_match)]
    shotdata=team_plot_data[team_plot_data['type'].isin(shots+goals)]
    games1=df[df['matchId']==matchId_giver(selected_match)]
    hlogo = Image.open("Logos/" + team_plot_data["home_team"].unique()[0] + ".png")
    alogo = Image.open("Logos/" + team_plot_data["away_team"].unique()[0] + ".png")
    mreport=match_report(team_plot_data, shotdata, hlogo, alogo, games1)
    st.pyplot(mreport)







