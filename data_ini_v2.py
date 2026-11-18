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



# fcts 3.1 & 3.2


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
    # liste_dfs -> liste python qui contien les tableaux filtres
    # liste_noms_dfs -> liste des noms pour la legende
    # normaliser -> booleen, True quand les histograms sont normaliser pour que laire totale soit egale a 1
    
    # si normaliser est True alors norme recoit du texte sinon elle est none
    if normaliser:
        norme = "probability density"
        unite_y = "Densité de probabilité"
    else:
        norme = None
        unite_y = "Nombre d'évènements"

    # boucle qui parcours les differentes variables
    for nom in noms_branches:

        # initialisation dune figure vide 
        fig = go.Figure()

        # boucle qui parcours les tableaux et leurs noms en meme temps
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



# fcts 3.3


def fit_donnees(donnees, obs, model):
    # je transforme la serie pandas en donnees compatibles avec zfit
    data = zfit.Data(data=donnees.to_numpy(), obs=obs)
    # nll -> log-vraisemblance negative etendue (cest ce que zfit va minimiser)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    # minimizer -> algorithme Minuit qui cherche le minimum
    minimizer = zfit.minimize.Minuit()
    # lancement du fit et mise a jour des parametres trouves
    result = minimizer.minimize(nll).update_params()
    # calcul des erreurs statistiques sur chaque parametre
    result.hesse()
    return result


# fit de la masse du D0 avec une gaussienne et une droite
def fit_masse(df, branche="D0_MM", min_fit=1816, max_fit=1914, nom_param=""):
    # on garde que les evenements qui sont entre min_fit et max_fit
    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    # obs -> espace de definition pour zfit
    obs = zfit.Space(branche, min_fit, max_fit)

    # parametres libres du modele (valeur de depart, min, max)
    # nom_param -> sert a rajouter un texte au nom pour pas que zfit plante quand on fait une boucle cool
    mu = zfit.Parameter(f"mu{nom_param}", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter(f"sigma{nom_param}", 8.0, 1.0, 25.0)
    c1 = zfit.Parameter(f"c1{nom_param}", 0.0, -1.0, 1.0)

    # nombre devenements de signal et de bruit
    n_sig = zfit.Parameter(f"n_sig{nom_param}", 0.5 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter(f"n_bkg{nom_param}", 0.5 * n_total, 0, n_total * 1.5)

    # sig_model -> gaussienne pour le pic du signal
    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    # bkg_model -> polynome de Chebyshev dordre 1 (une droite pour le bruit)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1], extended=n_bkg)
    # model -> somme des deux modeles
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    # lancement du fit avec la fonction dau dessus
    result = fit_donnees(donnees_masse, obs, model)

    # recuperation des valeurs trouvees par le fit pour calculer S et B (pour le 3.4)
    params = result.params
    mu_val = params[mu]["value"]
    sigma_val = params[sigma]["value"]
    S = params[n_sig]["value"]
    n_bkg_total = params[n_bkg]["value"]

    # fenetre_pic -> on prend la zone autour du pic entre mu - 3*sigma et mu + 3*sigma
    fenetre_pic = zfit.Space(branche, mu_val - 3 * sigma_val, mu_val + 3 * sigma_val)
    # integration de la droite du bruit juste sous le pic pour savoir combien de bruit est cache dedans
    # .numpy()[0] -> recupere juste le nombre decimal de zfit
    fraction_bruit_sous_pic = bkg_model.integrate(limits=fenetre_pic).numpy()[0]
    B = n_bkg_total * fraction_bruit_sous_pic

    return result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B



# fcts 3.4


# teste plusieurs coupures sur une variable et trace la FOM pour trouver la meilleure
def FOM_variable(df, nom_variable, liste_seuils, type_coupure="superieur", suffixe_fichier=""):
    # liste_seuils -> liste des valeurs a tester
    # type_coupure -> "superieur" pour garder (var > seuil) ou "inferieur" pour garder (var < seuil)

    liste_fom = []
    meilleure_fom = 0.0
    meilleur_seuil = liste_seuils[0]

    # boucle qui teste chaque seuil de la liste un par un
    for i in range(len(liste_seuils)):
        seuil = liste_seuils[i]

        if type_coupure == "superieur":
            df_test = df[df[nom_variable] > seuil]
        else:
            df_test = df[df[nom_variable] < seuil]

        # on refait le fit de masse sur les donnees filtrees pour avoir S et B
        result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(
            df_test, nom_param=f"_{nom_variable}{suffixe_fichier}_{i}"
        )

        # calcul de la FOM avec la formule du poly
        fom = S / np.sqrt(S + B)
        liste_fom.append(fom)

        # si la FOM est plus grande que lancienne meilleure on la garde
        if fom > meilleure_fom:
            meilleure_fom = fom
            meilleur_seuil = seuil

    # trace de la courbe FOM avec plotly cool
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=liste_seuils, # valeurs des seuils sur laxe x
        y=liste_fom, # valeurs de la FOM sur laxe y
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
    # sauvegarde automatique en html
    fig.write_html(f"{dossier_sortie}/FOM_{nom_variable}{suffixe_fichier}.html")

    return meilleur_seuil, meilleure_fom



# fcts 3.5


# fit avec une seule exponentielle (pour le 3.5.1 et le bruit du 3.5.2)
def fit_duree_vie_simple(df, branche="D0_TAU", min_tau=0.0004, max_tau=0.004, nom_param=""):
    # on part de 0.0004 ns pour eviter la coupure du detecteur au debut
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    # dans zfit lexponentielle cest exp(lam * t) donc lam = -1 / tau (vers -2400 ns^-1)
    lam = zfit.Parameter(f"lam{nom_param}", -2400.0, -15000.0, -100.0)
    n_evts = zfit.Parameter(f"n_evts{nom_param}", n_total, 0, n_total * 1.5)

    # modele exponentiel etendu
    model = zfit.pdf.Exponential(obs=obs, lam=lam, extended=n_evts)

    # on recycle la fonction de fit dau dessus
    result = fit_donnees(donnees_tau, obs, model)

    return result, model, donnees_tau, lam, n_evts


# fit avec 2 exponentielles : le signal libre + le bruit bloque
def fit_duree_vie_complet(df, lam_bkg_fixe, branche="D0_TAU", min_tau=0.0004, max_tau=0.004):
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    # lam_sig -> pente libre pour le signal
    lam_sig = zfit.Parameter("lam_sig_final", -2400.0, -15000.0, -100.0)
    # floating=False -> bloque la pente du bruit a la valeur trouvee sur les bandes laterales
    lam_bkg = zfit.Parameter("lam_bkg_final", lam_bkg_fixe, floating=False)

    n_sig = zfit.Parameter("n_sig_tau", 0.8 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_tau", 0.2 * n_total, 0, n_total * 1.5)

    # somme de lexponentielle du signal et de celle du bruit
    sig_model = zfit.pdf.Exponential(obs=obs, lam=lam_sig, extended=n_sig)
    bkg_model = zfit.pdf.Exponential(obs=obs, lam=lam_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    result = fit_donnees(donnees_tau, obs, model)

    return result, model, bkg_model, donnees_tau, lam_sig, n_sig, n_bkg



# exe 3.1  3.2


df_total = charge_donnees("data/MasterclassData_2012_all.root")

# list(df_total.columns) -> recupere tout les entetes des branches D0_MM ect.
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

# on lui donne les tableaux et les noms
dessine_branche_multiple(tableaux_a_comparer, noms_courbes, variables_a_tracer)

print(f"Tracés Plotly terminés et enregistrés dans : {dossier_sortie}")



# exe 3.3


# selection complete avec les coupures PID et geometriques
cond_dira = df_total["D0_DIRA_OWNPV"] > 0.9998
cond_ip = df_total["D0_MINIPCHI2"] < 9
masque_complet = cond_kaon & cond_pion & cond_dira & cond_ip
df_propre = df_total[masque_complet]

# lancement du fit sur les donnees nettoye
result, model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(df_propre)
print(result)

# je recupere les valeurs et les erreurs trouvees par le fit
params = result.params
mu_val = params[mu]["value"]
mu_err = params[mu]["hesse"]["error"]
n_sig_val = params[n_sig]["value"]
n_bkg_val = params[n_bkg]["value"]

print("Masse du D0 mesurée :", round(mu_val, 2), "+/-", round(mu_err, 2), "MeV/c²")
print("Nombre de signal :", int(n_sig_val), "| Nombre de bruit :", int(n_bkg_val))

# decoupage en 100 barres entre 1816 et 1914
nbins = 100
min_fit = 1816
max_fit = 1914
largeur_bin = (max_fit - min_fit) / nbins

# np.histogram -> calcule le nombre devenements dans chaque barre
comptes, bords = np.histogram(donnees_masse, bins=nbins, range=(min_fit, max_fit))

# boucle pour trouver le milieu de chaque barre et son erreur (racine de N)
centres = []
erreurs_data = []
for i in range(nbins):
    milieu = (bords[i] + bords[i + 1]) / 2
    centres.append(milieu)
    # si la barre est vide on met 1.0 pour eviter de diviser par zero apres
    if comptes[i] > 0:
        erreurs_data.append(np.sqrt(comptes[i]))
    else:
        erreurs_data.append(1.0)

centres = np.array(centres)
erreurs_data = np.array(erreurs_data)

# 500 points pour tracer une courbe bien lisse
x_lisse = np.linspace(min_fit, max_fit, 500)
y_total_lisse = model.pdf(x_lisse).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
y_bkg_lisse = bkg_model.pdf(x_lisse).numpy() * n_bkg_val * largeur_bin

# nouvelle figure pour le fit de masse
fig_fit = go.Figure()
fig_fit.add_trace(go.Scatter(
    x=centres,
    y=comptes,
    mode="markers",
    name="Données sélectionnées",
    error_y=dict(type="data", array=erreurs_data, visible=True), # barres derreur sur les points
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

# calcul des pulls dans chaque barre (question 3.3.4)
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
fig_pulls.add_hline(y=0, line_color="white", line_width=1) # ligne a 0
fig_pulls.add_hline(y=2, line_color="gray", line_dash="dash") # limite +2 sigma
fig_pulls.add_hline(y=-2, line_color="gray", line_dash="dash") # limite -2 sigma
fig_pulls.update_layout(
    title="Pulls du fit de masse du D⁰",
    title_x=0.5,
    xaxis_title="Masse D⁰ [MeV/c²]",
    yaxis_title="(Modèle - Données) / σ",
    template="plotly_dark"
)
fig_pulls.write_html(f"{dossier_sortie}/pulls_masse_D0.html")

# calcul du chi2 reduit (question 3.3.5)
n_params_flottants = 5 # mu, sigma, c1, n_sig, n_bkg
ndf = nbins - n_params_flottants # nombre de degres de liberte
chi2_reduit = somme_pulls_carre / ndf
print("Chi2 réduit :", round(chi2_reduit, 3), "(pour", ndf, "degrés de liberté)")



# exe 3.4


# 1. on cherche le meilleur seuil pour D0_DIRA_OWNPV avec les coupures PID deja mises (question 3.4.3 & 3.4.4)
seuils_dira = [0.9990, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]
opt_dira, fom_dira = FOM_variable(df_selectionne, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur")
print("Meilleur seuil D0_DIRA_OWNPV (avec PID) :", opt_dira, "| FOM max :", round(fom_dira, 2))

# 2. pareil pour D0_MINIPCHI2 apres avoir applique le meilleur DIRA
df_apres_dira = df_selectionne[df_selectionne["D0_DIRA_OWNPV"] > opt_dira]
seuils_ip = [25, 20, 16, 12, 9, 6, 4]
opt_ip, fom_ip = FOM_variable(df_apres_dira, "D0_MINIPCHI2", seuils_ip, type_coupure="inferieur")
print("Meilleur seuil D0_MINIPCHI2 (avec PID) :", opt_ip, "| FOM max :", round(fom_ip, 2))

# 3. test sur df_total sans aucune coupure PID au depart (question 3.4.5)
opt_dira_sans_pid, fom_sans_pid = FOM_variable(
    df_total, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", suffixe_fichier="_sans_PID"
)
print("Meilleur seuil D0_DIRA_OWNPV (SANS PID) :", opt_dira_sans_pid, "| FOM max :", round(fom_sans_pid, 2))



# exe 3.5


# application des meilleures coupures trouvees au 3.4
masque_optimal = cond_kaon & cond_pion & (df_total["D0_DIRA_OWNPV"] > opt_dira) & (df_total["D0_MINIPCHI2"] < opt_ip)
df_optimal = df_total[masque_optimal]

# separation entre la zone du pic et les bandes laterales (ou il y a que du bruit)
df_tau_pic = df_optimal[(df_optimal["D0_MM"] >= 1850) & (df_optimal["D0_MM"] <= 1882)]
df_tau_bruit = df_optimal[((df_optimal["D0_MM"] >= 1800) & (df_optimal["D0_MM"] <= 1835)) |
                          ((df_optimal["D0_MM"] >= 1895) & (df_optimal["D0_MM"] <= 1930))]

min_tau = 0.0004
max_tau = 0.004
nbins_tau = 60
largeur_bin_tau = (max_tau - min_tau) / nbins_tau

# --- 3.5.1 : fit avec 1 seule exponentielle sur le pic ---
res_simple, mod_simple, donnees_tau_pic, lam_simple, n_evts_simple = fit_duree_vie_simple(
    df_tau_pic, min_tau=min_tau, max_tau=max_tau, nom_param="_simple"
)

lam_s_val = res_simple.params[lam_simple]["value"]
lam_s_err = res_simple.params[lam_simple]["hesse"]["error"]
n_s_val = res_simple.params[n_evts_simple]["value"]

# conversion tau = -1 / lam en ns puis * 1000 pour lavoir en ps
tau_simple_ps = (-1.0 / lam_s_val) * 1000.0
err_tau_simple_ps = (lam_s_err / (lam_s_val ** 2)) * 1000.0
print("\n[3.5.1] Duree de vie (1 exp) :", round(tau_simple_ps, 4), "+/-", round(err_tau_simple_ps, 4), "ps")

# calcul du chi2 reduit et trace plotly
comptes_tau, bords_tau = np.histogram(donnees_tau_pic, bins=nbins_tau, range=(min_tau, max_tau))
centres_tau = []
erreurs_tau = []
for i in range(nbins_tau):
    centres_tau.append((bords_tau[i] + bords_tau[i + 1]) / 2)
    # meme securite que pour la masse si un bin est vide
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

# --- 3.5.2 : on mesure la pente du bruit sur les bandes laterales ---
res_bkg, mod_bkg_tau, donnees_tau_bkg, lam_bkg, n_evts_bkg = fit_duree_vie_simple(
    df_tau_bruit, min_tau=min_tau, max_tau=max_tau, nom_param="_bkg"
)
lam_bkg_val = res_bkg.params[lam_bkg]["value"]
print("[3.5.2] Pente lam du bruit de fond :", round(lam_bkg_val, 1), "ns^-1")

# --- 3.5.3 : fit final avec signal + bruit fixe ---
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