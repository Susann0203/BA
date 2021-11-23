import math
import random as rdm
from typing import List, Any

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import networkx as nx
import datetime as dt
import time

def read_data():
    # print("Bitte geben Sie den vollstaendigen Dateipfad der csv-Datei ein:")
    # path = input()
    path = "M:\\Dokumente\\Studium\\Info\\SS21\\BachelorArbeit\\x264_pervolution_measurements_bin.csv"
    influences = pd.read_csv(path, delimiter=";")
    #influences = influences.loc[influences["revision"]=="2.4.38"]
    return influences

def remove_colinearity(df): # koennte lange dauern
    # courtesy by johannes
    # remove columns with identical values (dead features or mandatory features)
    nunique = df.nunique()
    mandatory_or_dead = nunique[nunique == 1].index.values

    df = df.drop(columns=mandatory_or_dead)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df_configs = df
    alternative_ft = []
    alternative_ft_names = []
    group_candidates = {}

    for i, col in enumerate(df_configs.columns):
        filter_on = df_configs[col] == 1
        filter_off = df_configs[col] == 0
        group_candidates[col] = []
        for other_col in df_configs.columns:
            if other_col != col:
                values_if_col_on = df_configs[filter_on][other_col].unique()
                if len(values_if_col_on) == 1 and values_if_col_on[0] == 0:
                    # other feature is always off if col feature is on
                    group_candidates[col].append(other_col)

    G = nx.Graph()
    for ft, alternative_candidates in group_candidates.items():
        for candidate in alternative_candidates:
            if ft in group_candidates[candidate]:
                G.add_edge(ft, candidate)

    cliques_remaining = True
    while cliques_remaining:
        cliques_remaining = False
        cliques = nx.find_cliques(G)
        for clique in cliques:
            # check if exactly one col is 1 in each row
            sums_per_row = df_configs[clique].sum(axis=1).unique()
            if len(sums_per_row) == 1 and sums_per_row[0] == 1.0:
                delete_ft = sorted(clique)[0]
                alternative_ft_names.append(delete_ft)
                df_configs.drop(delete_ft, inplace=True, axis=1)
                for c in clique:
                    G.remove_node(c)
                cliques_remaining = True
                break
    # print(f"colnames in df after remove colinearity: {df.columns}")
    return df

def sort_data(influences):
    # sortiere csv-Spalten nach binaeren & numerischen Features sowie gemessenen Werten
    non_booleans = []
    for label, content in influences.items():
        for i in content:
            # wenn Feature in jeder Zeile 0 oder 1 stehen hat, ist es binaer, sonst nicht
            if i != 0 and i != 1:
                non_booleans.append(label)
                # print(f"{label} is not boolean")
                break

    booleans = [i for i, c in influences.items() if i not in non_booleans]
    # print("Bitte geben Sie die Anzahl der Spalten ein, die Messwerte beinhalten")
    # anzahl_perfs = int(input())
    # print("Bitte geben Sie die Namen der Messwerte ein, so wie sie in der csv-Datei stehen."
    #      "\nBestätigen Sie jede Eingabe mit Enter!")
    perfs = ["performance", "size", "cpu"]
    # print(f"\nBooleans: {booleans}")
    return booleans, non_booleans, perfs

def choose_first_samples(influences):
    # waehlt zufaellig min. 20 Samples oder 1/20 aller Kombinationen
    # Startmenge kann geaendert werden, indem choose_first_samples ersetzt/ ueberschrieben wird
    rdm.seed(dt.datetime.now())
    startSamples = []
    # i in range(50) falls Math.ceil(influences.shape[0]/20) < 50
    #x = math.ceil(influences.shape[0] / 40.0) if math.ceil(influences.shape[0] / 20.0) >= 50 else 50
    #print(f"x= {x}")
    x = 20
    for i in range(x):
        a = np.random.choice(influences.index)
        startSamples.append(influences.loc[a])
        influences = influences.drop(a)
        # print(influences.shape)
        # print(influences.iloc[a])
    return startSamples

def split_start_sample(startSapmle, bool_names, perf_names):
    # teile Startsample in Werte der binaeren Features und zugehoerige Performances
    booleans = []
    perfs = []
    # print(bool_names)
    for i in range(len(startSapmle)):
        new_b = []
        for a in bool_names:
            new_b.append(startSapmle[i][a])
        booleans.append(new_b)
        perfs.append(startSapmle[i][perf_names[0]])
    return booleans, perfs

def loop(influences, test_data, booleans, perf_names):
    # influences = Tabelle mit allen Trainings Featurekombis inkl. Performances
    # booleans = Liste mit Namen aller binaeren Features die noch vorhanden sind!
    # print(booleans)
    # print(influences.columns[0])
    coefs = []
    influences = remove_colinearity(influences)
    samples = choose_first_samples(influences)
    booleans2, perfs = split_start_sample(samples, booleans, perf_names) #Trainingsdaten
    test_bools = test_data[booleans]
    test_perfs = test_data[perf_names[0]]
    # print(booleans2)
    coefs_mean = None
    coefs_std = None
    lmodel = lm.LinearRegression()
    new_model, new_f_names, all_f_names = train_model(lmodel, booleans2, perfs, 0, 0, influences.columns)
    # Form all_f_names = [["feature1", "feature2"], ["feature1"], ["feature1", "feature3"],...]
    # print(f"new_model.coef_[0]: {new_model.coef_[0]}")
    coefs.append(new_model.coef_[0])
    influences, new_samples, coefs_mean, coefs_std = choose_next_samples(influences, coefs)
    samples.extend(new_samples)
    prediction = test_model_prediction(new_model, test_bools, test_perfs, booleans)
    ende_laufzeit_berechnung = time.time()
    #show(coefs_mean, coefs_std, all_f_names)  # zeige Mittelwerte und Stdabw.
    return ende_laufzeit_berechnung

def train_model(lmodel, start_sample_bools, start_sample_perfs, coefs_mean, coefs_std, colnames):
    start_sample_bools = np.array(start_sample_bools)   #start_sample_bools = training_sample
    start_sample_perfs = np.array(start_sample_perfs).reshape((-1, 1))
    poly = pre.PolynomialFeatures(degree=2, interaction_only=True) #Interaktion von 2Features ist vorerst ausreichend,
    # da auch fuer groessere Anzahl Features zu exponentiellem Wachstum der Anzahl der Interaktionen fuehrt
    X = poly.fit_transform(start_sample_bools)
    feature_names = colnames
    # print(f"feature_names: {feature_names}")
    feature_names = poly.get_feature_names(feature_names)
    # print("Size of Sample: " + str(start_sample_bools.shape))
    # print("Size of X: " + str(X.shape))
    # print(X)
    # print(start_sample_perfs.shape)
    # print("List of Bools: " + str(start_sample_bools))
    lmodel.fit(X, start_sample_perfs)
    # print("lmodel coef")
    # print(lmodel.coef_)
    scoore = lmodel.score(X, start_sample_perfs)
    # print(f"scoore: {scoore}")
    coefs_value = np.quantile(np.abs(scoore),0.85)
    # print(f"coefs_value: {coefs_value}")
    idx = np.where(np.abs(scoore) > coefs_value)[0]
    f_names = ['root'] + [feature_names[f] for f in idx]
    # print("coefs: ", idx)
    return lmodel, f_names, feature_names

def choose_next_samples(influences, coefs):
    # sucht die nächste Kombi, die die größte Ungenauigkeit aufweist
    coefs = np.vstack(coefs)
    # print(f"coefs: {coefs},len of 1 element: {len(coefs[0])}")
    coefs_mean = np.mean(coefs, axis=0) # berechne Mittelwerte
    coefs_std = np.std(coefs, axis=0)   # berechne Stdabw
    tmpinfls = []
    # print(f"Influences in choose_next_sample: {influences}")
    # print(f"coefs_std: {coefs_std}")
    new_configs, influences = find_according_konfigs(influences)
    tmpinfls.append(new_configs)
    return influences, tmpinfls, coefs_mean, coefs_std

def find_according_konfigs(influences):
    tmp1 = influences
    candidates1 = list(set(tmp1.index.values))
    new_configs = []
    for i in range(20):
        if len(candidates1) == 0:
            break
        if len(candidates1) > 0:
            new_config = np.random.choice(candidates1)
            new_configs.append(influences.loc[new_config])
    # print(f"new_configs: {new_configs}")
    return new_configs, influences

def test_model_prediction(model, test_bools, test_perfs, bool_names):
    # print(f"test_data: {test_data[bool_names]}")
    test_bools = test_bools.to_numpy()
    poly = pre.PolynomialFeatures(degree=2, interaction_only=True)
    X = poly.fit_transform(test_bools)
    feature_names = poly.get_feature_names(bool_names)
    prediction = model.predict(X)
    # print("Prediction:")
    # print(prediction)
    # print(f"shape of prediction: {prediction.shape}")
    # print(f"feature_names: {len(feature_names)}")
    # print(f"test_perfs: {type(test_perfs[0])}")
    # print(f"shape of X: {X.shape}")
    print(model.score(X, test_perfs))
    return model.score(X, test_perfs)

def show(lmodel, std_s, feature_names):
    # lmodel = means der einzelnen Features
    # std_s = stdabw der Features
    # feature_names = Namen aller Features
    coefs_value = np.quantile(np.abs(lmodel),0.85)
    idx = np.where(np.abs(lmodel) > coefs_value)[0]
    plt.figure(figsize=(16, 5))
    # print(idx)
    # print(len(feature_names))
    plt.bar([feature_names[f] for f in idx], [lmodel[f] for f in idx])
    _ = plt.xticks(rotation=90)
    stds_value = np.quantile(np.abs(std_s),0.85)
    # print(f"stds_value: {stds_value}")
    idx2 = np.where(np.abs(std_s) > std_s)[0]
    plt.bar([feature_names[f] for f in idx2], [std_s[f] for f in idx2], width=[0.5 for i in idx2])
    plt.title(type(lmodel))
    plt.show()

if __name__ == "__main__":
    start = time.time()
    infl = read_data()
    infl2 = remove_colinearity(infl)    # Nummerierung der Features anpassen bzw abfragen
    # da dann evtl verschiebungen ggue. der urspruenglichen csv auftreten
    print(infl2.shape)
    train_infl, test_infl = ms.train_test_split(infl2, train_size=0.1)
    # print(f"train_infl:\n{train_infl}")
    bool_names, non_bool_names, perf_names = sort_data(train_infl)
    ende = loop(train_infl, test_infl, bool_names, perf_names)

    print('{:5.3f}s'.format(ende-start))
