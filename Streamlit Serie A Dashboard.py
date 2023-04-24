import streamlit as st
import mplsoccer as mps
import pandas as pd
import plotly.express as px
from PIL import Image
from urllib.request import urlopen
import plotly.graph_objects as go
from mplsoccer import add_image
from ast import literal_eval
import numpy as np
from matplotlib.colors import to_rgba
st.title("Serie A Dashboard 2022/23")
shots=['MissedShots','SavedShot','ShotOnPost']
goals=['Goal']
def_actions=['BallRecovery','Clearance','Tackle','Challenge','Interception','BlockedPass']
takeon=["TakeOn"]
touches=["BallTouch"]
events=pd.read_csv("C:/Users/Rahul/FA masterclass/data/Seriea_Events.csv",low_memory=False)
df=pd.read_csv("C:/Users/Rahul/FA masterclass/data/Serie A Fixtures.csv")
eventsdf=pd.merge(events,df[["home_team","away_team","matchId"]],on="matchId",how="left")
xT = pd.read_csv("C:/Users/Rahul/OneDrive/Pictures/Juve vs Inter/xT_Grid.csv", header=None)
xT = np.array(xT)
xT_rows, xT_cols = xT.shape
passes=eventsdf[eventsdf["type"]=="Pass"]
passes['x1_bin'] = pd.cut(passes['x'], bins=xT_cols, labels=False)
passes['y1_bin'] = pd.cut(passes['y'], bins=xT_rows, labels=False)
passes['x2_bin'] = pd.cut(passes['endX'], bins=xT_cols, labels=False)
passes['y2_bin'] = pd.cut(passes['endY'], bins=xT_rows, labels=False)
passes['start_zone_value'] = passes[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
passes['end_zone_value'] = passes[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
passes['xT'] = passes['end_zone_value'] - passes['start_zone_value']
options=["Player Report","Match Report"]
st.sidebar.header('Choose Viz Type')
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
    x=pd.DataFrame(literal_eval(test2.iloc[0]["qualifiers"]))
    list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
    list2=[]
    for i in range(len(list1)):
        plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[i])]["playerName"].unique().tolist()[0]
        list2.append(plyr)
    list3=[]
    for i in range(len(test2)):
        x=pd.DataFrame(literal_eval(test2.iloc[i]["qualifiers"]))
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
    x = pd.DataFrame(literal_eval(test2.iloc[0]["qualifiers"]))
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
        x = pd.DataFrame(literal_eval(test2.iloc[i]["qualifiers"]))
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
    x=pd.DataFrame(literal_eval(test2.iloc[0]["qualifiers"]))
    list1=str.split(x[x["type"]=="InvolvedPlayers"]["value"].tolist()[0],',',11)[:11]
    list2=[]
    for i in range(len(list1)):
        plyr=events_df[events_df["playerId"]==pd.to_numeric(list1[i])]["playerName"].unique().tolist()[0]
        list2.append(plyr)
    list3=[]
    for i in range(len(test2)):
        x=pd.DataFrame(literal_eval(test2.iloc[i]["qualifiers"]))
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
if(selected_viz_type=="Player Report"):
    players = eventsdf["playerName"].unique().tolist()
    players = [x for x in players if str(x) != 'nan']
    st.sidebar.header('Player Input Tab')
    selected_player = st.sidebar.selectbox('Player', players)
    hal = eventsdf[eventsdf["playerName"] == selected_player]
    hal1 = passes[passes["playerName"] == selected_player]
    pitch = mps.VerticalPitch(line_color="black", pitch_color="#FFF8DC", line_zorder=2, pitch_type='opta')
    fig, axs = pitch.grid(nrows=1, ncols=3, title_height=0.2, axis=False, grid_width=0.9, figheight=17)
    fig.set_facecolor("#FFF8DC")
    pitch.arrows(hal1[hal1["xT"] > 0]["x"], hal1[hal1["xT"] > 0]["y"],
                 hal1[hal1["xT"] > 0]["endX"], hal1[hal1["xT"] > 0]["endY"], color="black", ax=axs["pitch"][0])
    pitch.arrows(hal[hal["assist"] == True]["x"], hal[hal["assist"] == True]["y"],
                 hal[hal["assist"] == True]["endX"], hal[hal["assist"] == True]["endY"], color="red",
                 ax=axs["pitch"][0], label="Assist")
    pitch.scatter(hal[hal["type"].isin(takeon)]["x"], hal[hal["type"].isin(takeon)]["y"], color="red",
                  edgecolor="black", s=300, ax=axs["pitch"][1], label="Take Ons")
    pitch.scatter(hal[hal["type"].isin(shots)]["x"], hal[hal["type"].isin(shots)]["y"], color="#FFF8DC",
                  edgecolor="black", s=300, ax=axs["pitch"][1], label="Shot")
    pitch.scatter(hal[hal["type"].isin(goals)]["x"], hal[hal["type"].isin(goals)]["y"], marker="football", s=300,
                  ax=axs["pitch"][1], label="Goal")
    pitch.scatter(hal[hal["type"].isin(def_actions)]["x"], hal[hal["type"].isin(def_actions)]["y"], marker='*', s=300,
                  ax=axs["pitch"][1], label="Def Actions")
    bs_heatmap1 = pitch.bin_statistic_positional(hal["x"], hal["y"], positional='full', statistic='count',
                                                 normalize=True)
    hm = pitch.heatmap_positional(bs_heatmap1, ax=axs["pitch"][2], cmap='Reds', edgecolor="black")
    pitch.label_heatmap(bs_heatmap1, color="white", ha="center", va="center", backgroundcolor='0.5', ax=axs["pitch"][2],
                        str_format='{:.0%}', fontsize=30)
    axs["pitch"][0].set_title(selected_player + " Passes", fontsize=30)
    axs["pitch"][1].set_title(selected_player + " Shots & Def Actions", fontsize=30)
    axs["pitch"][2].set_title(selected_player + " Touch Zones Map", fontsize=30)
    axs["pitch"][1].legend(edgecolor='black', fontsize=12, loc='upper left', handlelength=5)
    axs["pitch"][0].legend(edgecolor='black', fontsize=12, loc='upper left', handlelength=5)
    title = axs['title'].text(0.5, 1, selected_player + ' : Actions in 22/23 (made by @Rahulvn5)',
                              ha='center', va='center', fontsize=40, color='black', weight="bold")

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
    print(team_plot_data)
    avgdfh = avg_dfgetter(team_plot_data, "h")
    avgdfa = avg_dfgetter(team_plot_data, "a")
    pmergeh = pass_network_dfgetter(team_plot_data, "h")
    pmergea = pass_network_dfgetter(team_plot_data, "a")
    lineuph = lineup_getter(team_plot_data, "h")
    lineupa = lineup_getter(team_plot_data, "a")
    passes = team_plot_data[team_plot_data["type"] == "Pass"]
    eventsh = passes[passes["h_a"] == "h"]
    eventsa = passes[passes["h_a"] == "a"]
    pitch = mps.VerticalPitch(line_color="black", pitch_color="#FFF8DC", line_zorder=2, pitch_type='opta')
    fig1, axs = pitch.grid(nrows=2, ncols=3, title_height=0.1, axis=False, grid_width=0.9, figheight=30)
    fig1.set_facecolor("#FFF8DC")
    MIN_TRANSPARENCY = 0.3

    pitch.arrows(eventsh[eventsh["type"] == "Pass"]["x"], eventsh[eventsh["type"] == "Pass"]["y"],
                 eventsh[eventsh["type"] == "Pass"]["endX"], eventsh[eventsh["type"] == "Pass"]["endY"],
                 color="#BF3EFF", ax=axs["pitch"][0][0])
    pitch.arrows(eventsa[eventsa["type"] == "Pass"]["x"], eventsa[eventsa["type"] == "Pass"]["y"],
                 eventsa[eventsa["type"] == "Pass"]["endX"], eventsa[eventsa["type"] == "Pass"]["endY"],
                 color="#BF3EFF", ax=axs["pitch"][1][0])
    bins = (6, 4)
    bs_heatmap1 = pitch.bin_statistic(eventsh[eventsh["type"] == "Pass"]["x"], eventsh[eventsh["type"] == "Pass"]["y"],
                                      statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap1, ax=axs["pitch"][0][1], cmap='Blues')
    # plot the pass flow map with a single color ('black') and length of the arrow (5)
    fm = pitch.flow(eventsh[eventsh["type"] == "Pass"]["x"], eventsh[eventsh["type"] == "Pass"]["y"],
                    eventsh[eventsh["type"] == "Pass"]["endX"], eventsh[eventsh["type"] == "Pass"]["endY"],
                    color='black', arrow_type='same',
                    arrow_length=5, bins=bins, ax=axs["pitch"][0][1])
    bs_heatmap2 = pitch.bin_statistic(eventsa[eventsa["type"] == "Pass"]["x"], eventsa[eventsa["type"] == "Pass"]["y"],
                                      statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap2, ax=axs["pitch"][1][1], cmap='Blues')
    # plot the pass flow map with a single color ('black') and length of the arrow (5)
    fm = pitch.flow(eventsa[eventsa["type"] == "Pass"]["x"], eventsa[eventsa["type"] == "Pass"]["y"],
                    eventsa[eventsa["type"] == "Pass"]["endX"], eventsa[eventsa["type"] == "Pass"]["endY"],
                    color='black', arrow_type='same',
                    arrow_length=5, bins=bins, ax=axs["pitch"][1][1])
    ws = pmergeh["width"].values.tolist()
    ws = [i * 10 for i in ws]
    pitch.lines(pmergeh["x"], pmergeh["y"], pmergeh["endx"], pmergeh["endy"], lw=ws, color=pmergeh['alpha'], zorder=1,
                ax=axs["pitch"][0][2])
    ws1 = pmergea["width"].values.tolist()
    ws1 = [i * 10 for i in ws1]
    pitch.lines(pmergea["x"], pmergea["y"], pmergea["endx"], pmergea["endy"], lw=ws1, color=pmergea['alpha'], zorder=1,
                ax=axs["pitch"][1][2])
    for i in range(len(lineuph)):
        pitch.scatter(avgdfh[avgdfh["playerName"] == lineuph[i]]["x"],
                      avgdfh[avgdfh["playerName"] == lineuph[i]]["y"], s=1000, color="red", ax=axs["pitch"][0][2])
        try:
            pitch.annotate(lineuph[i].split(" ", 1)[1], xy=(
                avgdfh[avgdfh["playerName"] == lineuph[i]]["x"] - 2, avgdfh[avgdfh["playerName"] == lineuph[i]]["y"]),
                           c='black', va='center', ha='center', ax=axs["pitch"][0][2], fontsize=22)
        except:
            pitch.annotate(lineuph[i], xy=(
                avgdfh[avgdfh["playerName"] == lineuph[i]]["x"] - 2, avgdfh[avgdfh["playerName"] == lineuph[i]]["y"]),
                           c='black', va='center', ha='center', ax=axs["pitch"][0][2], fontsize=22)
    for i in range(len(lineupa)):
        pitch.scatter(avgdfa[avgdfa["playerName"] == lineupa[i]]["x"],
                      avgdfa[avgdfa["playerName"] == lineupa[i]]["y"], s=1000, color="red", ax=axs["pitch"][1][2])
        try:
            pitch.annotate(lineupa[i].split(" ", 1)[1], xy=(
                avgdfa[avgdfa["playerName"] == lineupa[i]]["x"] - 3, avgdfa[avgdfa["playerName"] == lineupa[i]]["y"]),
                           c='black', va='center', ha='center', ax=axs["pitch"][1][2], fontsize=22)
        except:
            pitch.annotate(lineupa[i], xy=(
                avgdfa[avgdfa["playerName"] == lineupa[i]]["x"] - 3, avgdfa[avgdfa["playerName"] == lineupa[i]]["y"]),
                           c='black', va='center', ha='center', ax=axs["pitch"][1][2], fontsize=22)
    axs["pitch"][0][0].set_title(team_plot_data["home_team"].unique()[0] + " Passes", color="black", fontsize=30)
    axs["pitch"][0][1].set_title(team_plot_data["home_team"].unique()[0] + " Pass Flow Plot", color="black",
                                 fontsize=30)
    axs["pitch"][0][2].set_title(team_plot_data["home_team"].unique()[0] + " \nPlayers Avg. Positions & Network",
                                 color="black", fontsize=30)
    axs["pitch"][1][0].set_title(team_plot_data["away_team"].unique()[0] + " Passes", color="black", fontsize=30)
    axs["pitch"][1][1].set_title(team_plot_data["away_team"].unique()[0] + " Pass Flow Plot", color="black",
                                 fontsize=30)
    axs["pitch"][1][2].set_title(team_plot_data["away_team"].unique()[0] + " Players Avg. Positions & Network",
                                 color="black", fontsize=30)
    title = axs['title'].text(0.5, 1,
                              team_plot_data["home_team"].unique()[0] + ' vs ' + team_plot_data["away_team"].unique()[
                                  0] + ' Match Report 22/23 (made by @Rahulvn5)',
                              ha='center', va='center', fontsize=40, color='black', weight="bold")
    hlogo = Image.open("C:/Users/Rahul/Logos/Serie A teams/" + team_plot_data["home_team"].unique()[0] + ".png")
    alogo = Image.open("C:/Users/Rahul/Logos/Serie A teams/" + team_plot_data["away_team"].unique()[0] + ".png")
    add_image(hlogo, fig1, left=0.2, bottom=0.865, width=0.2, height=0.05)
    add_image(alogo, fig1, left=0.6, bottom=0.865, width=0.2, height=0.05)
    st.pyplot(fig1, axs)
    