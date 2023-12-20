from flask import Flask, json, request, jsonify
from flask_cors import CORS
import requests
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd

api = Flask(__name__)
CORS(api)

routes = {
    "api_endpoint": "api.riotgames.com",
    "nearest_cluster": "europe",        # now that's it's Riot ID, this is the recommendation
    "league": "/lol/league/v4/",
    "match": "/lol/match/v5/matches/",
    "summoner": "/lol/summoner/v4/summoners/by-name/",
    "account": "/riot/account/v1/accounts/by-riot-id/"
}

@api.route('/api/test', methods=['POST', 'OPTIONS'])
def get_test():
    if request.method == 'OPTIONS':
        print("CORS set-up")
        return json.dumps({"Hello":"World"})
    print(">>> Got request:")
    data = request.json
    print(data["summonerName"])
    print(request.headers.get("X-Riot-Token"))
    return json.dumps({"Hello":"World"})

@api.route('/api/matches', methods=['POST','OPTIONS'])
def get_matches():
    if request.method == 'OPTIONS':
        print("CORS set-up")
        return json.dumps({"Hello":"World"})

    print(">>> Got request with JSON:")
    req = request.json
    print(req)

    summoner_name   = req["summonerName"]
    tag_line        = req["tagline"]
    region_short    = req["regionShort"]
    region_long     = req["regionLong"]
    start           = req["startFrom"]
    apiKey          = request.headers["X-Riot-Token"]
    res_data        = { "puuid": "",
                        "matchIds": [],
                        "matches": [],
                        "matchTimelines": [],
                        "predictions": []   }
    
    print(f"API Key: {apiKey}")

    # first get the player's PUUID
    # loc = f"https://{region_short}.{routes['api_endpoint']}{routes['summoner']}{summoner_name}"
    loc = f"https://{routes['nearest_cluster']}.{routes['api_endpoint']}{routes['account']}{summoner_name}/{tag_line}"
    
    # sending get request and saving the response as response object
    # print("Sending request to Riot API to get PUUID")
    r = requests.get(url = loc, headers = {"X-Riot-Token": apiKey})
    if r.status_code == 200:
        print(f"Got PUUID: {r.json()['puuid']}")
        data = r.json()
        res_data["puuid"] = data["puuid"]

        # now get the match IDs
        loc = f"https://{region_long}.{routes['api_endpoint']}{routes['match']}by-puuid/{res_data['puuid']}/ids?type=ranked&start={start}&count=20"
        # print(loc)
        # print("Sending request to Riot API to get match IDs")
        r = requests.get(url = loc, headers = {"X-Riot-Token": apiKey})

        if r.status_code == 200:
            print("Got it")
            data = r.json()
            res_data["matchIds"] = data

            # finally get the actual matches
            matches = []
            matchTimelines = []
            predictions = []
            for id in tqdm(res_data["matchIds"]):
                # print(f"Sending request to Riot API to get match: {id}")
                loc = f"https://{region_long}.{routes['api_endpoint']}{routes['match']}{id}"
                r = requests.get(url = loc, headers = {"X-Riot-Token": apiKey})
                
                if r.status_code == 200:
                    # print("Got it")
                    data = r.json()
                    matches.append(data["info"])

                    # print(f"Sending request to Riot API to get match timeline: {id}")
                    loc = f"https://{region_long}.{routes['api_endpoint']}{routes['match']}{id}/timeline"
                    r = requests.get(url = loc, headers = {"X-Riot-Token": apiKey})
                    
                    if r.status_code == 200:
                        # print("Got it")
                        data = r.json()
                        matchTimelines.append(data["info"])
                        if matches[-1]["gameMode"] != 'CLASSIC':
                            predictions.append("Not CLASSIC")
                        else:
                            preds = get_predictions(data["info"])
                            predictions.append(preds)
                    elif r.status_code == 429:
                        return "Too many requests. Wait a while before trying again :(", 429
                    else:
                        print (r)
                        return "Something went wrong when getting a match's timelines :(", 500
                elif r.status_code == 429:
                    return "Too many requests. Wait a while before trying again :(", 429
                else:
                    print (r)
                    return "Something went wrong when getting a match's info :(", 500
                
            res_data["matches"] = matches
            res_data["matchTimelines"] = matchTimelines
            res_data["predictions"] = predictions
            # with open('data-sample.json', 'w') as f:
            #     json.dump(res_data, f, indent=4)
            return json.dumps(res_data), 200
        elif r.status_code == 429:
            print(r.status_code)
            return "Too many requests. Wait a while before trying again :(", 429
        else:
            print (r.status_code)
            return "Something went wrong when getting the match list (IDs) :(", 500
    
    elif r.status_code == 429:
        return "Too many requests. Wait a while before trying again :(", 429
    elif r.status_code == 403:
        return "403 - Unathorized. Your API key might no longer be valid :(", 403
    elif r.status_code == 404:
        return "404 - Not found. You might be looking for a summoner that doesn't exist in the region you specified :(", 403
    else:
        print (r.content)
        return "Something went wrong :(", 500


def get_predictions(match_timeline):
    # creating an empty list to save the data in, and then create the dataframe
    to_save = []

    # info for total objectives taken by either sides
    blue_turrets    = 0
    blue_barons     = 0
    blue_drakes     = 0
    blue_kills      = 0
    blue_inhibs     = 0
    blue_heralds    = 0
    total_blue_elders = 0

    red_turrets    = 0
    red_barons     = 0
    red_drakes     = 0
    red_kills      = 0
    red_inhibs     = 0
    red_heralds    = 0
    total_red_elders = 0
    
    # info for first major objectives taken by either side (boolean flag)
    blue_first_blood    = 0.0
    blue_first_herald   = 0.0
    blue_first_drake    = 0.0
    blue_first_baron    = 0.0
    blue_first_inhib    = 0.0
    blue_first_turret   = 0.0
    # to have soul, the team should have four drakes total not counting elders
    blue_got_soul = 0.0

    red_first_blood    = 0.0
    red_first_herald   = 0.0
    red_first_drake    = 0.0
    red_first_baron    = 0.0
    red_first_inhib    = 0.0
    red_first_turret   = 0.0
    # to have soul, the team should have four drakes total not counting elders
    red_got_soul = 0.0

    for m in match_timeline["frames"]:

        # info for stats and objectives taken by each player in either side - init
        blue_player_kills   = [0, 0, 0, 0, 0]
        blue_player_xp      = []
        blue_player_gold    = []
        blue_player_dmg     = []
        blue_player_vs      = [0, 0, 0, 0, 0]   # can't get vision score from timeline :(
        red_player_kills   = [0, 0, 0, 0, 0]
        red_player_xp      = []
        red_player_gold    = []
        red_player_dmg     = []
        red_player_vs      = [0, 0, 0, 0, 0]

        # loop through players to add above data
        for idx in ['1','2','3','4','5']:       # blue side
            blue_player_xp.append(m["participantFrames"][idx]["xp"])
            blue_player_gold.append(m["participantFrames"][idx]["totalGold"])
            blue_player_dmg.append(m["participantFrames"][idx]["damageStats"]["totalDamageDoneToChampions"])
        for idx in ['6','7','8','9','10']:       # red side
            red_player_xp.append(m["participantFrames"][idx]["xp"])
            red_player_gold.append(m["participantFrames"][idx]["totalGold"])
            red_player_dmg.append(m["participantFrames"][idx]["damageStats"]["totalDamageDoneToChampions"])

        for ev in m["events"]:
            if ev["type"] == "CHAMPION_KILL":
                blue_k = ev["killerId"] < 6 and ev["killerId"] > 0
                if blue_k:                      # blue got kill
                    blue_kills                          += 1
                    blue_player_kills[ev["killerId"]-1] += 1 
                    if red_first_blood < 1 and blue_first_blood < 1:  # first blood
                        red_first_blood     = 1.0
                        blue_first_blood    = 0.0
                elif ev["killerId"] > 0:        # red got kill (not execution)
                    red_kills                          += 1
                    red_player_kills[ev["killerId"]-5-1] += 1
                    if red_first_blood < 1 and blue_first_blood < 1:  # first blood
                        red_first_blood     = 0.0
                        blue_first_blood    = 1.0

            elif ev["type"] == "ELITE_MONSTER_KILL":
                if ev["monsterType"] == "DRAGON":
                    if ev["killerTeamId"] == 100:   # blue got drake
                        blue_drakes += 1
                        if red_first_drake < 1 and blue_first_drake < 1:
                            red_first_drake = 0.0
                            blue_first_drake = 1.0
                    else:                          # red got drake
                        red_drakes += 1
                        if red_first_drake < 1 and blue_first_drake < 1:
                            red_first_drake = 1.0
                            blue_first_drake = 0.0
                
                elif ev["monsterType"] == "ELDER_DRAGON":
                    if ev["killerTeamId"] == 100:   # blue got drake
                        total_blue_elders += 1
                    else:                          # red got drake
                        total_red_elders += 1

                elif ev["monsterType"] == "RIFTHERALD":
                    if ev["killerTeamId"] == 100:   # blue got drake
                        blue_heralds += 1
                        if red_first_herald < 1 and blue_first_herald < 1:
                            red_first_herald = 0.0
                            blue_first_herald = 1.0
                    else:                          # red got herald
                        red_heralds += 1
                        if red_first_herald < 1 and blue_first_herald < 1:
                            red_first_herald = 1.0
                            blue_first_herald = 0.0

                elif ev["monsterType"] == "BARON_NASHOR":
                    if ev["killerTeamId"] == 100:   # blue got drake
                        blue_barons += 1
                        if red_first_baron < 1 and blue_first_baron < 1:
                            red_first_baron = 0.0
                            blue_first_baron = 1.0
                    else:                          # red got baron
                        red_barons += 1
                        if red_first_baron < 1 and blue_first_baron < 1:
                            red_first_baron = 1.0
                            blue_first_baron = 0.0
            
            elif ev["type"] == "DRAGON_SOUL_GIVEN":
                if ev["teamId"] == 100:
                    blue_got_soul   = 1.0
                    red_got_soul    = 0.0
                else:
                    blue_got_soul   = 0.0
                    red_got_soul    = 1.0

            elif ev["type"] == "BUILDING_KILL":
                if ev["buildingType"] == "TOWER_BUILDING":
                    if ev["teamId"] != 100:       # blue got turret
                        blue_turrets += 1
                        if red_first_turret < 1 and blue_first_turret < 1:
                            red_first_turret = 0.0
                            blue_first_turret = 1.0
                    else:                          # red got turret
                        red_turrets += 1
                        if red_first_turret < 1 and blue_first_turret < 1:
                            red_first_turret = 1.0
                            blue_first_turret = 0.0

                elif ev["buildingType"] == "INHIBITOR_BUILDING":
                    if ev["teamId"] != 100:       # blue got inhib
                        blue_inhibs += 1
                        if red_first_inhib < 1 and blue_first_inhib < 1:
                            red_first_inhib = 0.0
                            blue_first_inhib = 1.0
                    else:                          # red got inhib
                        red_inhibs += 1
                        if red_first_inhib < 1 and blue_first_inhib < 1:
                            red_first_inhib = 1.0
                            blue_first_inhib = 0.0       

        # calculations for actual features
        total_blue_kills    = blue_kills / (red_kills + blue_kills) if (red_kills + blue_kills) != 0 else 0
        total_red_kills     = red_kills / (red_kills + blue_kills) if (red_kills + blue_kills) != 0 else 0

        total_blue_barons   = blue_barons / (red_barons + blue_barons) if (red_barons + blue_barons) != 0 else 0
        total_red_barons    = red_barons / (red_barons + blue_barons) if (red_barons + blue_barons) != 0 else 0

        total_blue_turrets  = blue_turrets / (red_turrets + blue_turrets) if (red_turrets + blue_turrets) != 0 else 0
        total_red_turrets   = red_turrets / (red_turrets + blue_turrets) if (red_turrets + blue_turrets) != 0 else 0

        total_blue_drakes   = blue_drakes / (red_drakes + blue_drakes) if (red_drakes + blue_drakes) != 0 else 0
        total_red_drakes    = red_drakes / (red_drakes + blue_drakes) if (red_drakes + blue_drakes) != 0 else 0

        total_blue_inhibs   = blue_inhibs / (red_inhibs + blue_inhibs) if (red_inhibs + blue_inhibs) != 0 else 0
        total_red_inhibs    = red_inhibs / (red_inhibs + blue_inhibs) if (red_inhibs + blue_inhibs) != 0 else 0

        total_blue_heralds  = blue_heralds / (red_heralds + blue_heralds) if (red_heralds + blue_heralds) != 0 else 0
        total_red_heralds   = red_heralds / (red_heralds + blue_heralds) if (red_heralds + blue_heralds) != 0 else 0

        total_blue_gold     = np.sum(blue_player_gold) / (np.sum(blue_player_gold) + np.sum(red_player_gold))
        total_red_gold      = np.sum(red_player_gold) / (np.sum(blue_player_gold) + np.sum(red_player_gold))

        total_blue_elders   = total_blue_elders / (total_blue_elders + total_red_elders) if total_blue_elders != 0 else 0
        total_red_elders    = total_red_elders / (total_blue_elders + total_red_elders) if total_red_elders != 0 else 0

        total_blue_vs     = np.sum(blue_player_vs) / (np.sum(blue_player_vs) + np.sum(red_player_vs)) if (np.sum(blue_player_vs) + np.sum(red_player_vs)) != 0 else 0
        total_red_vs      = np.sum(red_player_vs) / (np.sum(blue_player_vs) + np.sum(red_player_vs)) if (np.sum(blue_player_vs) + np.sum(red_player_vs)) != 0 else 0

        # calculations of indivudal player-based medians
        med_blue_kills  = [0,0,0,0,0]
        med_red_kills   = [0,0,0,0,0]
        for k in range(5):
            if (blue_player_kills[k] + red_player_kills[k]) != 0:
                med_blue_kills[k]   = blue_player_kills[k] / (blue_player_kills[k] + red_player_kills[k])
                med_red_kills[k]    = red_player_kills[k] / (blue_player_kills[k] + red_player_kills[k])
        med_blue_kills  = np.median(med_blue_kills)
        med_red_kills   = np.median(med_red_kills)

        med_blue_xp  = [0,0,0,0,0]
        med_red_xp   = [0,0,0,0,0]
        for k in range(5):
            if (blue_player_xp[k] + red_player_xp[k]) != 0:
                med_blue_xp[k]   = blue_player_xp[k] / (blue_player_xp[k] + red_player_xp[k])
                med_red_xp[k]    = red_player_xp[k] / (blue_player_xp[k] + red_player_xp[k])
        med_blue_xp  = np.median(med_blue_xp)
        med_red_xp   = np.median(med_red_xp)

        med_blue_gold  = [0,0,0,0,0]
        med_red_gold   = [0,0,0,0,0]
        for k in range(5):
            if (blue_player_gold[k] + red_player_gold[k]) != 0:
                med_blue_gold[k]   = blue_player_gold[k] / (blue_player_gold[k] + red_player_gold[k])
                med_red_gold[k]    = red_player_gold[k] / (blue_player_gold[k] + red_player_gold[k])
        med_blue_gold  = np.median(med_blue_gold)
        med_red_gold   = np.median(med_red_gold)

        med_blue_dmg  = [0,0,0,0,0]
        med_red_dmg   = [0,0,0,0,0]
        for k in range(5):
            if (blue_player_dmg[k] + red_player_dmg[k]) != 0:
                med_blue_dmg[k]   = blue_player_dmg[k] / (blue_player_dmg[k] + red_player_dmg[k])
                med_red_dmg[k]    = red_player_dmg[k] / (blue_player_dmg[k] + red_player_dmg[k])
        med_blue_dmg  = np.median(med_blue_dmg)
        med_red_dmg   = np.median(med_red_dmg)

        med_blue_vs  = [0,0,0,0,0]
        med_red_vs   = [0,0,0,0,0]
        for k in range(5):
            if (blue_player_vs[k] + red_player_vs[k]) != 0:
                med_blue_vs[k]   = blue_player_vs[k] / (blue_player_vs[k] + red_player_vs[k])
                med_red_vs[k]    = red_player_vs[k] / (blue_player_vs[k] + red_player_vs[k])
        med_blue_vs  = np.median(med_blue_vs)
        med_red_vs   = np.median(med_red_vs)

        row = [total_blue_barons,total_blue_drakes,total_blue_heralds,total_blue_inhibs,total_blue_kills,total_blue_turrets,
            blue_first_blood, blue_first_herald, blue_first_drake, blue_first_baron, blue_first_inhib, blue_first_turret, total_blue_vs,
            total_blue_gold, med_blue_kills, med_blue_xp, med_blue_gold, med_blue_dmg, blue_got_soul, total_blue_elders, med_blue_vs,
            total_red_barons, total_red_drakes, total_red_heralds, total_red_inhibs, total_red_kills, total_red_turrets,
            red_first_blood, red_first_herald, red_first_drake, red_first_baron, red_first_inhib, red_first_turret, total_red_vs,
            total_red_gold, med_red_kills, med_red_xp, med_red_gold, med_red_dmg, red_got_soul, total_red_elders, med_red_vs
            ]
        
        to_save.append(row)

    cols = ["total_blue_barons", "total_blue_drakes", "total_blue_heralds", "total_blue_inhibs", "total_blue_kills", "total_blue_turrets",
        "blue_first_blood", "blue_first_herald", "blue_first_drake", "blue_first_baron", "blue_first_inhib", "blue_first_turret", "total_blue_vs",
        "total_blue_gold", "med_blue_kills", "med_blue_xp", "med_blue_gold", "med_blue_dmg", "blue_got_soul", "total_blue_elders", "med_blue_vs",
        "total_red_barons", "total_red_drakes", "total_red_heralds", "total_red_inhibs", "total_red_kills", "total_red_turrets",
        "red_first_blood", "red_first_herald", "red_first_drake", "red_first_baron", "red_first_inhib", "red_first_turret", "total_red_vs",
        "total_red_gold", "med_red_kills", "med_red_xp", "med_red_gold", "med_red_dmg", "red_got_soul", "total_red_elders", "med_red_vs"
        ]
    df = pd.DataFrame(to_save, columns=cols).fillna(0).to_numpy()

    with open('model.pkl', 'rb') as f:
        svc = pickle.load(f)
    
    # make predictions
    prob = np.array(svc.predict_proba(df))

    return prob.tolist()


if __name__ == '__main__':
    api.run(debug=True)