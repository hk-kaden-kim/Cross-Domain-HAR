import matplotlib.pyplot as plt
color_plt = plt.cm.tab10.colors

VARIANTS = [
            "base", "var-C", "var-U", "var-CU",
            "var-CUA-s", "var-CUA-r", "var-CUA-a", 
            "var-CUA-sr", "var-CUA-ra", "var-CUA-sa", 
            "var-CUA-sra"
           ]

THRESHOLD = {
            "base":-1, "var-C":-1, "var-U":0.6395594849642504, "var-CU":0.16982040019441205,
            "var-CUA-s":0.14381345004871413, "var-CUA-r":0.27748399225421777, "var-CUA-a":0.25648232789130476, 
            "var-CUA-sr":0.14751688498120277, "var-CUA-ra":0.21599618538237744, "var-CUA-sa":0.18240330630242105, 
            "var-CUA-sra":0.11794567144367274
           }
VARIANTS_COLOR = {'base':color_plt[0] , # base:
                  'var-C':color_plt[1] , # var-C: 
                  'var-CU':color_plt[2] , # var-CU: 
                  'var-CUA':color_plt[3] } # var-CUA: 

DS_NAME = [
    '_base', '_base_unkn',    # Source DATASET (unkn='Paal' + 'Commuting')
    'adl','forthtrace','mendeleydaily',     # Target DATASET
    'oppo','selfback','wristppg',       # Target DATASET without any Unknown labels
    'ichi14','newcastle',       # Target DATASET with only one label
    ]
DS_LABEL = [
    'source', 'source+unknown',    # Source DATASET
    'target-1.1','target-1.2','target-1.3',     # Target DATASET
    'target-2.1','target-2.2','target-2.3',       # Target DATASET without any Unknown labels
    'target-3.1','target-3.2',       # Target DATASET with only one label
    ]

DS_COLOR = ["#000000", 
             "#5e904e", "#aed0a0", "#02531d", # Green Pair Preference
             "#256676", "#1ceaf9", "#4ea6dc", # Blue Pair Preference
             "#9f3b60", "#ef8ead"]  # Red Pair Preference

LABEL_COLORS={
    'sit-stand': color_plt[0], # 
    'sleep': color_plt[1], # 
    'mixed': color_plt[2], # 
    'walking': color_plt[3], # 
    'vehicle': color_plt[4], # 
    'bicycling': color_plt[5], # 
    'unknown': "#000000", # #D0312D
}

CUA_STYLES = {
    "var-CUA-s":{'marker':'o', 'color': color_plt[4]},
    "var-CUA-r":{'marker':'^', 'color': color_plt[4]},
    "var-CUA-a":{'marker':'s', 'color': color_plt[4]},
    "var-CUA-sr":{'marker':'o', 'color': color_plt[5]},
    "var-CUA-ra":{'marker':'^', 'color': color_plt[5]},
    "var-CUA-sa":{'marker':'s', 'color': color_plt[5]},
    "var-CUA-sra":{'marker':'*', 'color': color_plt[6]},
}
# ['o','^','S']



