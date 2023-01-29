# bivouac_ia_avance
Projet IA Avancé de la majeure Santé de EPITA avec le projet Bivouac du FHU Mosaic (APHP Beaujon)

Ce projet vise à évaluer les performances de plusieurs modèles ResNet, en fonction de la taille des tuiles prises sur des pièces opératoires.

Nous avons principalement travaillé sur un calculateur nommé Romeo sur lequelle nous avons pu utiliser des GPU P100 en parrallèles.
Cela explique la structure de notre code, avec des scripts bash appelé avec une commande de type : *requete* + <fichier .sh>, permettant d'envoyer de décrire la tache à effectuer.

Le dossier "weights" contient les modèle préentrainés", "script" les scripts à donner à slurm pour lancer les training, "src" les fichiers pythons de training de chaque ResNet pour chaque taille de tuiles
