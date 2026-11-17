import os
import uproot
import numpy as np
import pandas as pd
import zfit
import plotly.graph_objects as go
import plotly.express as px

# variable pour stocker le nom du dossier dans lequel je vais sauvegarder les differents histograms
# "histograms_plots" est un chemin relatif, python va ce debrouiller
dossier_sortie = "histograms_plots"


# ==========================================
# FONCTIONS SECTIONS 3.1 & 3.2
# ==========================================

def charge_donnees(chemin_fichier, branches=None):
    # Je recupere la ou les branches du fichier root au format pandas data frame
    fichier = uproot.open(chemin_fichier)
    arbre = fichier["DecayTree"]
    tableau = arbre.arrays(branches, library="pd")
    return tableau

# f"{} -> f-string (format string) sert a insere directement la valeur dune variable dans du texte cool

# tracage dun ou des histogramms
def dessine_branche(df, noms_branches, nbins=169, dossier="histograms_plots"):

    # boucle qui parcourt noms_branches element par element
    for nom in noms_branches:
        # figure interactive de plotly
        fig = px.histogram(
            df, # table de donnees
            x=nom, # definition de la colonne du tableau df a mettre sur laxe x
            nbins=nbins, # nombre de barres
            title=f"Distribution de {nom}", # titre de lhistogram
            template="plotly_dark", # Theme sombre de plotly
            opacity=0.85 # opacite de lhistogram
        )
        fig.update_layout(
            xaxis_title=nom, # titre de laxe x
            yaxis_title="Nombre d'évènements", # titre de laxe y
            bargap=0.03, # distances entre les barres
            title_x=0.5 # position du titre au centre du graphique
        )
        # sauvegarde de lhistogram en fichier html, automatique gere par plotly cool
        fig.write_html(f"{dossier}/hist_{nom}.html")


# tracage de plusieurs histograms sur un graphique
def dessine_branche_multiple(liste_dfs, liste_noms_dfs, noms_branches, nbins=169, normaliser=True, dossier="histograms_plots"):
    # liste_dfs -> liste des tableaux de donnees a comparer
    # liste_noms_dfs -> liste des noms associes a chaque tableau (pour la legende)
    # normaliser -> booleen, True quand les histograms sont normaliser pour que laire totale soit egale a 1
    
    # si normaliser est True alors norme recoit du texte sinon elle est none
    if normaliser:
        norme = "probability density"
        unite_y = "Densité de probabilité"
    else:
        norme = None
        unite_y = "Nombre d'évènements"

    # boucle qui parcours les differentes variables (branches)
    for nom in noms_branches:

        # initialisation dune figure vide 
        fig = go.Figure()

        # boucle avec indice i pour parcourir les tableaux et leurs noms en meme temps
        for i in range(len(liste_dfs)):
            df_actuel = liste_dfs[i]
            label_actuel = liste_noms_dfs[i]

            # ajoute un graphique sur la figure en cours
            fig.add_trace(go.Histogram(
                x=df_actuel[nom], # extraction de la colonne pour la metre sur laxe des x
                name=label_actuel,
                nbinsx=nbins,
                histnorm=norme, # application de la norme
                opacity=0.65
            ))

        fig.update_layout(
            title=f"Comparaison : {nom}", # titre du graphique
            title_x=0.5, 
            xaxis_title=nom, 
            yaxis_title=unite_y,
            barmode="overlay", # superposition des histogrammes par plotly
            template="plotly_dark",
            bargap=0.02
        )

        fig.write_html(f"{dossier}/superposition_{nom}.html") # enregistrement entant que fichier html


# ==========================================
# FONCTIONS SECTION 3.3
# ==========================================

def fit_donnees(donnees, obs, model):
    # conversion de la serie pandas au format zfit
    data = zfit.Data(data=donnees.to_numpy(), obs=obs)
    # definition de la log-vraisemblance negative etendue
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    # algorithme de minimisation Minuit
    minimizer = zfit.minimize.Minuit()
    # lancement de la minimisation et mise a jour des parametres
    result = minimizer.minimize(nll).update_params()
    # calcul des erreurs statistiques sur les parametres
    result.hesse()
    return result


def fit_masse(df, branche="D0_MM", min_fit=1815, max_fit=1915, nom_param=""):
    # on garde les evenements compris entre min_fit et max_fit
    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    # espace de definition de zfit
    obs = zfit.Space(branche, min_fit, max_fit)

    # parametres libres des modeles (valeur initiale, min, max)
    # nom_param sert a eviter que zfit bloque quand on appelle fit_masse en boucle
    mu = zfit.Parameter(f"mu{nom_param}", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter(f"sigma{nom_param}", 8.0, 1.0, 25.0)
    c1 = zfit.Parameter(f"c1{nom_param}", 0.0, -1.0, 1.0)

    # nombre d'evenements de signal et de bruit de fond
    n_sig = zfit.Parameter(f"n_sig{nom_param}", 0.5 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter(f"n_bkg{nom_param}", 0.5 * n_total, 0, n_total * 1.5)

    # modele du signal : Gaussienne
    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    # modele du bruit de fond : polynome de Chebyshev d'ordre 1 (droite)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1], extended=n_bkg)
    # somme des deux modeles
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    # lancement du fit avec notre fonction generique
    result = fit_donnees(donnees_masse, obs, model)

    # calcul de S et B sous le pic (fenetre +/- 3 sigma autour de mu) pour la section 3.4
    params = result.params
    mu_val = params[mu]["value"]
    sigma_val = params[sigma]["value"]
    S = params[n_sig]["value"]
    n_bkg_total = params[n_bkg]["value"]

    # on integre la densite de probabilite du bruit uniquement sous le pic [mu - 3*sigma, mu + 3*sigma]
    fenetre_pic = zfit.Space(branche, mu_val - 3 * sigma_val, mu_val + 3 * sigma_val)
    fraction_bruit_sous_pic = bkg_model.integrate(limits=fenetre_pic).numpy()[0]
    B = n_bkg_total * fraction_bruit_sous_pic

    return result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B


# ==========================================
# FONCTIONS SECTION 3.4
# ==========================================

def FOM_variable(df, nom_variable, liste_seuils, type_coupure="superieur", suffixe_fichier=""):
    # liste_seuils -> liste des valeurs de coupures a tester
    # type_coupure -> "superieur" pour garder (var > seuil) ou "inferieur" pour garder (var < seuil)

    liste_fom = []
    meilleure_fom = 0.0
    meilleur_seuil = liste_seuils[0]

    # boucle qui teste chaque seuil un par un
    for i in range(len(liste_seuils)):
        seuil = liste_seuils[i]

        if type_coupure == "superieur":
            df_test = df[df[nom_variable] > seuil]
        else:
            df_test = df[df[nom_variable] < seuil]

        # on fait le fit de masse sur cet echantillon filtre pour recuperer S et B
        result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(
            df_test, nom_param=f"_{nom_variable}{suffixe_fichier}_{i}"
        )

        # calcul de la Figure de Merite S / sqrt(S + B)
        fom = S / np.sqrt(S + B)
        liste_fom.append(fom)

        # on garde en memoire le seuil qui donne la plus grande FOM
        if fom > meilleure_fom:
            meilleure_fom = fom
            meilleur_seuil = seuil

    # tracage de la courbe FOM en fonction du seuil avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=liste_seuils,
        y=liste_fom,
        mode="lines+markers",
        name=f"FOM ({nom_variable})",
        line=dict(color="gold", width=2.5),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"Optimisation de la sélection : FOM pour {nom_variable}{suffixe_fichier}",
        title_x=0.5,
        xaxis_title=f"Seuil sur {nom_variable}",
        yaxis_title="FOM = S / √(S + B)",
        template="plotly_dark"
    )
    fig.write_html(f"{dossier_sortie}/FOM_{nom_variable}{suffixe_fichier}.html")

    return meilleur_seuil, meilleure_fom


# ==========================================
# FONCTIONS SECTION 3.5
# ==========================================

def fit_duree_vie_simple(df, branche="D0_TAU", min_tau=0.0004, max_tau=0.004, nom_param=""):
    # fit avec une seule exponentielle (utilise pour la question 3.5.1 et le bruit en 3.5.2)
    # on commence a 0.0004 ns pour eviter le seuil de coupure du detecteur aux tres petits temps
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    # dans zfit, Exponential est exp(lam * t), donc lam = -1 / tau (environ -2400 ns^-1)
    lam = zfit.Parameter(f"lam{nom_param}", -2400.0, -15000.0, -100.0)
    n_evts = zfit.Parameter(f"n_evts{nom_param}", n_total, 0, n_total * 1.5)

    model = zfit.pdf.Exponential(obs=obs, lam=lam, extended=n_evts)

    # on reutilise notre fonction generique de fit
    result = fit_donnees(donnees_tau, obs, model)

    return result, model, donnees_tau, lam, n_evts


def fit_duree_vie_complet(df, lam_bkg_fixe, branche="D0_TAU", min_tau=0.0004, max_tau=0.004):
    # fit avec deux exponentielles : signal (libre) + bruit de fond (pente fixee)
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    # parametre libre pour le signal
    lam_sig = zfit.Parameter("lam_sig_final", -2400.0, -15000.0, -100.0)
    # parametre bloque (floating=False) pour le bruit de fond
    lam_bkg = zfit.Parameter("lam_bkg_final", lam_bkg_fixe, floating=False)

    n_sig = zfit.Parameter("n_sig_tau", 0.8 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_tau", 0.2 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Exponential(obs=obs, lam=lam_sig, extended=n_sig)
    bkg_model = zfit.pdf.Exponential(obs=obs, lam=lam_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    result = fit_donnees(donnees_tau, obs, model)

    return result, model, bkg_model, donnees_tau, lam_sig, n_sig, n_bkg


# ==========================================
# EXECUTION SECTIONS 3.1 & 3.2
# ==========================================

df_total = charge_donnees("data/MasterclassData_2012_all.root")

# recuperation des noms de toutes les colonnes dans une liste simple
liste_branches = list(df_total.columns)
# visualisation des donnees
print("Branches disponibles :", liste_branches)

# creation des differents histogrammes individuellement
dessine_branche(df_total, liste_branches)

# selection simple sur lidentification des particules 
cond_kaon = df_total["Kplus_ProbNNk"] > 0.95
cond_pion = df_total["piminus_ProbNNpi"] > 0.8
masque_sel = cond_kaon & cond_pion

# application du masque sur les tableau 
df_selectionne = df_total[masque_sel]

# nouvelle figure
fig_sel = go.Figure()

fig_sel.add_trace(go.Histogram(
    x=df_total["D0_MM"], # le masque deja applique a partire des probabilite
    name="Avant sélection",
    nbinsx=120,
    histnorm="probability density",
    opacity=0.6
))
fig_sel.add_trace(go.Histogram(
    x=df_selectionne["D0_MM"],
    name="Après coupures ProbNN",
    nbinsx=120,
    histnorm="probability density",
    opacity=0.6
))
fig_sel.update_layout(
    title="Effet des coupures ProbNN sur le pic de masse du D⁰",
    title_x=0.5,
    xaxis_title="Masse D0",
    xaxis_range=[1800, 1930],
    yaxis_title="Densité de probabilité",
    barmode="overlay",
    template="plotly_dark",
    bargap=0.02
)
fig_sel.write_html(f"{dossier_sortie}/comparaison_selection.html")

# on prend que les evenements qui contienne la mesure de la masse de D
cond_pic = (df_total["D0_MM"] >= 1850) & (df_total["D0_MM"] <= 1882)
df_signal_region = df_total[cond_pic]

# on prend les mesures qui se situent en dehors de la mesure de la masse de D
cond_bande_gauche = (df_total["D0_MM"] >= 1814) & (df_total["D0_MM"] <= 1849)
cond_bande_droite = (df_total["D0_MM"] >= 1883) & (df_total["D0_MM"] <= 1914)
df_bruit_region = df_total[cond_bande_gauche | cond_bande_droite]

# Superposition Signal vs Bruit avec deux listes simples
tableaux_a_comparer = [df_signal_region, df_bruit_region]
noms_courbes = ["Région Pic (Signal dominant)", "Bandes Latérales (Bruit)"]
variables_a_tracer = ["D0_DIRA_OWNPV", "D0_MINIPCHI2", "Kplus_ProbNNk", "piminus_ProbNNpi"]

dessine_branche_multiple(tableaux_a_comparer, noms_courbes, variables_a_tracer)

print(f"Tracés Plotly terminés et enregistrés dans : {dossier_sortie}")


# ==========================================
# EXECUTION SECTION 3.3
# ==========================================

# selection complete avec les coupures PID et geometriques
cond_dira = df_total["D0_DIRA_OWNPV"] > 0.9998
cond_ip = df_total["D0_MINIPCHI2"] < 9
masque_complet = cond_kaon & cond_pion & cond_dira & cond_ip
df_propre = df_total[masque_complet]

# appel de la fonction de fit sur les donnees nettoyees
result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(df_propre)
print(result)

# lecture des valeurs et des erreurs trouvees par le fit
params = result.params
mu_val = params[mu]["value"]
mu_err = params[mu]["hesse"]["error"]
n_sig_val = params[n_sig]["value"]
n_bkg_val = params[n_bkg]["value"]

print("Masse du D0 mesurée :", round(mu_val, 2), "+/-", round(mu_err, 2), "MeV/c²")
print("Nombre de signal :", int(n_sig_val), "| Nombre de bruit :", int(n_bkg_val))

# creation de l'histogramme des donnees en 100 barres entre 1800 et 1930
nbins = 100
min_fit = 1816
max_fit = 1914
largeur_bin = (max_fit - min_fit) / nbins

comptes, bords = np.histogram(donnees_masse, bins=nbins, range=(min_fit, max_fit))

# calcul du centre et de l'incertitude statistique (racine de N) pour chaque barre
centres = []
erreurs_data = []
for i in range(nbins):
    milieu = (bords[i] + bords[i + 1]) / 2
    centres.append(milieu)
    if comptes[i] > 0:
        erreurs_data.append(np.sqrt(comptes[i]))
    else:
        erreurs_data.append(1.0)

centres = np.array(centres)
erreurs_data = np.array(erreurs_data)

# calcul des courbes continues du modele sur 500 points pour un beau trace
x_lisse = np.linspace(min_fit, max_fit, 500)
y_total_lisse = model.pdf(x_lisse).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
y_bkg_lisse = bkg_model.pdf(x_lisse).numpy() * n_bkg_val * largeur_bin

# graphique du fit de masse
fig_fit = go.Figure()
fig_fit.add_trace(go.Scatter(
    x=centres,
    y=comptes,
    mode="markers",
    name="Données sélectionnées",
    error_y=dict(type="data", array=erreurs_data, visible=True),
    marker=dict(color="cyan", size=5)
))
fig_fit.add_trace(go.Scatter(
    x=x_lisse,
    y=y_total_lisse,
    mode="lines",
    name=f"Fit zfit (M = {mu_val:.2f} ± {mu_err:.2f} MeV/c²)",
    line=dict(color="orange", width=2.5)
))
fig_fit.add_trace(go.Scatter(
    x=x_lisse,
    y=y_bkg_lisse,
    mode="lines",
    name="Bruit de fond (Chebyshev)",
    line=dict(color="red", width=2, dash="dash")
))
fig_fit.update_layout(
    title="Fit étendu (zfit) de la masse du D⁰",
    title_x=0.5,
    xaxis_title="Masse D⁰ [MeV/c²]",
    yaxis_title=f"Évènements / {largeur_bin:.2f} MeV/c²",
    template="plotly_dark"
)
fig_fit.write_html(f"{dossier_sortie}/fit_masse_D0.html")

# calcul des pulls dans chaque bin (Question 3.3.4)
prediction_bins = model.pdf(centres).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
pulls = []
somme_pulls_carre = 0.0

for i in range(nbins):
    pull_i = (prediction_bins[i] - comptes[i]) / erreurs_data[i]
    pulls.append(pull_i)
    somme_pulls_carre = somme_pulls_carre + pull_i ** 2

# trace des pulls
fig_pulls = go.Figure()
fig_pulls.add_trace(go.Bar(x=centres, y=pulls, name="Pulls", marker_color="mediumseagreen"))
fig_pulls.add_hline(y=0, line_color="white", line_width=1)
fig_pulls.add_hline(y=2, line_color="gray", line_dash="dash")
fig_pulls.add_hline(y=-2, line_color="gray", line_dash="dash")
fig_pulls.update_layout(
    title="Pulls du fit de masse du D⁰",
    title_x=0.5,
    xaxis_title="Masse D⁰ [MeV/c²]",
    yaxis_title="(Modèle - Données) / σ",
    template="plotly_dark"
)
fig_pulls.write_html(f"{dossier_sortie}/pulls_masse_D0.html")

# calcul du chi2 reduit (Question 3.3.5)
n_params_flottants = 5 # mu, sigma, c1, n_sig, n_bkg
ndf = nbins - n_params_flottants
chi2_reduit = somme_pulls_carre / ndf
print("Chi2 réduit :", round(chi2_reduit, 3), "(pour", ndf, "degrés de liberté)")


# ==========================================
# EXECUTION SECTION 3.4
# ==========================================

# 1. Optimisation de D0_DIRA_OWNPV avec la selection PID initiale (Question 3.4.3 & 3.4.4)
seuils_dira = [0.9990, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]
opt_dira, fom_dira = FOM_variable(df_selectionne, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur")
print("Meilleur seuil D0_DIRA_OWNPV (avec PID) :", opt_dira, "| FOM max :", round(fom_dira, 2))

# 2. Optimisation de D0_MINIPCHI2 apres avoir applique la coupure optimale sur DIRA
df_apres_dira = df_selectionne[df_selectionne["D0_DIRA_OWNPV"] > opt_dira]
seuils_ip = [25, 20, 16, 12, 9, 6, 4]
opt_ip, fom_ip = FOM_variable(df_apres_dira, "D0_MINIPCHI2", seuils_ip, type_coupure="inferieur")
print("Meilleur seuil D0_MINIPCHI2 (avec PID) :", opt_ip, "| FOM max :", round(fom_ip, 2))

# 3. Test SANS aucune selection PID initiale sur df_total (Question 3.4.5)
opt_dira_sans_pid, fom_sans_pid = FOM_variable(
    df_total, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", suffixe_fichier="_sans_PID"
)
print("Meilleur seuil D0_DIRA_OWNPV (SANS PID) :", opt_dira_sans_pid, "| FOM max :", round(fom_sans_pid, 2))


# ==========================================
# EXECUTION SECTION 3.5
# ==========================================

# On applique notre selection optimale obtenue a la section 3.4
masque_optimal = cond_kaon & cond_pion & (df_total["D0_DIRA_OWNPV"] > opt_dira) & (df_total["D0_MINIPCHI2"] < opt_ip)
df_optimal = df_total[masque_optimal]

# On separe la zone du pic (signal) et les bandes laterales (bruit de fond pur)
df_tau_pic = df_optimal[(df_optimal["D0_MM"] >= 1850) & (df_optimal["D0_MM"] <= 1882)]
df_tau_bruit = df_optimal[((df_optimal["D0_MM"] >= 1800) & (df_optimal["D0_MM"] <= 1835)) |
                          ((df_optimal["D0_MM"] >= 1895) & (df_optimal["D0_MM"] <= 1930))]

min_tau = 0.0004
max_tau = 0.004
nbins_tau = 60
largeur_bin_tau = (max_tau - min_tau) / nbins_tau

# --- 3.5.1 : Fit avec une simple exponentielle sur la region du pic ---
res_simple, mod_simple, donnees_tau_pic, lam_simple, n_evts_simple = fit_duree_vie_simple(
    df_tau_pic, min_tau=min_tau, max_tau=max_tau, nom_param="_simple"
)

lam_s_val = res_simple.params[lam_simple]["value"]
lam_s_err = res_simple.params[lam_simple]["hesse"]["error"]
n_s_val = res_simple.params[n_evts_simple]["value"]

# Conversion tau = -1 / lam (en nanosecondes, puis * 1000 pour avoir des picosecondes)
tau_simple_ps = (-1.0 / lam_s_val) * 1000.0
err_tau_simple_ps = (lam_s_err / (lam_s_val ** 2)) * 1000.0
print("\n[3.5.1] Duree de vie (1 exp) :", round(tau_simple_ps, 4), "+/-", round(err_tau_simple_ps, 4), "ps")

# Calcul du chi2 reduit et trace Plotly pour le fit simple
comptes_tau, bords_tau = np.histogram(donnees_tau_pic, bins=nbins_tau, range=(min_tau, max_tau))
centres_tau = []
erreurs_tau = []
for i in range(nbins_tau):
    centres_tau.append((bords_tau[i] + bords_tau[i + 1]) / 2)
    if comptes_tau[i] > 0:
        erreurs_tau.append(np.sqrt(comptes_tau[i]))
    else:
        erreurs_tau.append(1.0)
centres_tau = np.array(centres_tau)
erreurs_tau = np.array(erreurs_tau)

pred_simple = mod_simple.pdf(centres_tau).numpy() * n_s_val * largeur_bin_tau
somme_pulls_simple = 0.0
for i in range(nbins_tau):
    pull_i = (pred_simple[i] - comptes_tau[i]) / erreurs_tau[i]
    somme_pulls_simple = somme_pulls_simple + pull_i ** 2
print("[3.5.1] Chi2 reduit (1 exp) :", round(somme_pulls_simple / (nbins_tau - 2), 3))

x_tau_lisse = np.linspace(min_tau, max_tau, 500)
y_tau_simple_lisse = mod_simple.pdf(x_tau_lisse).numpy() * n_s_val * largeur_bin_tau

fig_tau1 = go.Figure()
fig_tau1.add_trace(go.Scatter(
    x=centres_tau, y=comptes_tau, mode="markers", name="Données région pic",
    error_y=dict(type="data", array=erreurs_tau, visible=True),
    marker=dict(color="cyan", size=5)
))
fig_tau1.add_trace(go.Scatter(
    x=x_tau_lisse, y=y_tau_simple_lisse, mode="lines",
    name=f"Fit 1 exp (τ = {tau_simple_ps:.4f} ± {err_tau_simple_ps:.4f} ps)",
    line=dict(color="orange", width=2.5)
))
fig_tau1.update_layout(
    title="Fit de la durée de vie du D⁰ (une seule exponentielle)",
    title_x=0.5,
    xaxis_title="Temps de vol D0_TAU [ns]",
    yaxis_title="Évènements",
    template="plotly_dark"
)
fig_tau1.write_html(f"{dossier_sortie}/fit_tau_simple.html")

# --- 3.5.2 : Modele du bruit de fond via les bandes laterales ---
res_bkg, mod_bkg_tau, donnees_tau_bkg, lam_bkg, n_evts_bkg = fit_duree_vie_simple(
    df_tau_bruit, min_tau=min_tau, max_tau=max_tau, nom_param="_bkg"
)
lam_bkg_val = res_bkg.params[lam_bkg]["value"]
print("[3.5.2] Pente lam du bruit de fond :", round(lam_bkg_val, 1), "ns^-1")

# --- 3.5.3 : Fit complet (Exponentielle Signal + Exponentielle Bruit fixee) ---
res_comp, mod_comp, bkg_comp, donnees_tau_pic, lam_sig, n_sig_tau, n_bkg_tau = fit_duree_vie_complet(
    df_tau_pic, lam_bkg_val, min_tau=min_tau, max_tau=max_tau
)

lam_sig_val = res_comp.params[lam_sig]["value"]
lam_sig_err = res_comp.params[lam_sig]["hesse"]["error"]
n_sig_t_val = res_comp.params[n_sig_tau]["value"]
n_bkg_t_val = res_comp.params[n_bkg_tau]["value"]

tau_final_ps = (-1.0 / lam_sig_val) * 1000.0
err_tau_final_ps = (lam_sig_err / (lam_sig_val ** 2)) * 1000.0
print("[3.5.3] Duree de vie finale (Signal + Bruit) :", round(tau_final_ps, 4), "+/-", round(err_tau_final_ps, 4), "ps")

pred_comp = mod_comp.pdf(centres_tau).numpy() * (n_sig_t_val + n_bkg_t_val) * largeur_bin_tau
somme_pulls_comp = 0.0
for i in range(nbins_tau):
    pull_i = (pred_comp[i] - comptes_tau[i]) / erreurs_tau[i]
    somme_pulls_comp = somme_pulls_comp + pull_i ** 2
print("[3.5.3] Chi2 reduit (Signal + Bruit) :", round(somme_pulls_comp / (nbins_tau - 3), 3))

y_tau_comp_lisse = mod_comp.pdf(x_tau_lisse).numpy() * (n_sig_t_val + n_bkg_t_val) * largeur_bin_tau
y_tau_bkg_lisse = bkg_comp.pdf(x_tau_lisse).numpy() * n_bkg_t_val * largeur_bin_tau

fig_tau2 = go.Figure()
fig_tau2.add_trace(go.Scatter(
    x=centres_tau, y=comptes_tau, mode="markers", name="Données région pic",
    error_y=dict(type="data", array=erreurs_tau, visible=True),
    marker=dict(color="cyan", size=5)
))
fig_tau2.add_trace(go.Scatter(
    x=x_tau_lisse, y=y_tau_comp_lisse, mode="lines",
    name=f"Fit total (τ = {tau_final_ps:.4f} ± {err_tau_final_ps:.4f} ps)",
    line=dict(color="orange", width=2.5)
))
fig_tau2.add_trace(go.Scatter(
    x=x_tau_lisse, y=y_tau_bkg_lisse, mode="lines",
    name="Contribution bruit de fond",
    line=dict(color="red", width=2, dash="dash")
))
fig_tau2.update_layout(
    title="Fit de la durée de vie du D⁰ (Signal + Bruit de fond)",
    title_x=0.5,
    xaxis_title="Temps de vol D0_TAU [ns]",
    yaxis_title="Évènements",
    template="plotly_dark"
)
fig_tau2.write_html(f"{dossier_sortie}/fit_tau_complet.html")


# ==========================================
# REPONSES TEXTE
# ==========================================

# 3.1 Preambule

# We can see that the Measured Mass of the D0 is contained in the branch D0_MM
# the lifetime of the D0 is contained in the branch D0_TAU
# and the transverse momentum of each particle is contained in the branches D0_PT, Kplus_PT, piminus_PT
# the probability of a kaon candidate being identified as pion or a kaon is stored in the branches Kplus_ProbNNpi and Kplus_ProbNNk
# the probability of a pion candidate being identified as pion or a kaon is stored in the branches piminus_ProbNNpi and piminus_ProbNNk

# 3.2 Selection

# The easiest data from which we can 'extract' the signal by a simple selection with as little noise as possible is the probability histograms of the kaons and the pions because of the probability peaks that are easy to spot.
# I think the best easy way to seperate the data with the noise and the important data is to check the full histograms and find a value on the Y axis where i will 'cut' the histogram.
# It comes from the cos of the theta angle, if we considere the conservation of momentum the angle between the primary vertex and the secondary vertex should be close to 0 for hte real D particle. so we can put aside all of the traces that have an angle that is superior to 0.