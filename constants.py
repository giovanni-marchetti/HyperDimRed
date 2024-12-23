gs_lf_tasks = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
fruits = ['apple','apricot']

chemical_features_r=["nCIR",
                     "ZM1",
                     "GNar",
                     "S1K",
                     "piPC08",
                     "MATS1v",
                     "MATS7v",
                     "GATS1v",
                     "Eig05_AEA(bo)",
                     "SM02_AEA(bo)",
                     "SM03_AEA(dm)",
                     "SM10_AEA(dm)",
                     "SM13_AEA(dm)",
                      "SpMin3_Bh(v)",
                     "RDF035v",
                     "G1m",
                     "G1v",
                     "G1e",
                     "G3s",
                     "R8u+",
                     "nRCOSR"]


keller_descriptors = ['0.1',
     '1.1',
     '2.1',
     '3.1',
     '4.1',
     '5.1',
     '6.1',
     '7.1',
     '8.1',
     '9.1',
     '10.1',
     '11.1',
     '12.1',
     '13.1',
     '14.1',
     '15.1',
     '16.1',
     '17.1',
     '18.1',
     '19.1',
     '20.1',
     '21.1']
sagar_descriptors = ['0.1',
     '1.1',
     '2.1',
     '3.1',
     '4.1',
     '5.1',
     '6.1',
     '7.1',
     '8.1',
     '9.1',
     '10.1',
     '11.1',
     '12.1',
     '13.1',
     '14.1'
     ]

CID_sagar = [126
,177
,196
,239
,240
,261
,325
,326
,356
,379
,460
,650
,660
,957
,1001
,1068
,1110
,1136
,2214
,2345
,2346
,2758
,2879
,2969
,3893
,4133
,5541
,5610
,5779
,5780
,5960
,6050
,6106
,6184
,6549
,6584
,6590
,6943
,6997
,6998
,7059
,7095
,7122
,7136
,7144
,7151
,7165
,7361
,7410
,7463
,7500
,7519
,7593
,7635
,7654
,7731
,7749
,7761
,7762
,7795
,7799
,7803
,7967
,7983
,8077
,8082
,8093
,8103
,8129
,8174
,8184
,8186
,8193
,8375
,8456
,8658
,8697
,8712
,8723
,8785
,8797
,8908
,8914
,9025
,9589
,9609
,9862
,10364
,10400
,10448
,10882
,10890
,10895
,10925
,11086
,11428
,11525
,11527
,11529
,11583
,11614
,11902
,11980
,12297
,12327
,12587
,12741
,12810
,12813
,14257
,14286
,14296
,14491
,14525
,15037
,17617
,18635
,18827
,21057
,21648
,22873
,23642
,24834
,26331
,27458
,31209
,31234
,31244
,31246
,31249
,31265
,31272
,32594
,36822
,61138
,61199
,61527
,61653
,61670
,61918
,62375
,62378
,62444
,62835
,62902
,82227
,89440
,170833
,235414
,439570
,440967
,444539
,520191
,637563
,637796
,638014
,3578033
,5315892
,5366244
,6999977
]

keller_tasks = ['Acid',
 'Ammonia',
 'Bakery',
 'Burnt',
 'Chemical',
 'Cool',
 'Decayed',
 'Familiarity',
 'Fishy',
 'Floral',
 'Fruity',
 'Garlic',
 'Grass',
 'Intensity',
 'Musky',
 'Pleasantness',
 'Sour',
 'Spicy',
 'Sweaty',
 'Sweet',
 'Warm',
 'Wood']

common_tasks = ['Fishy', 'Burnt', 'Sour','Decayed','Musky','Fruity','Sweaty','Cool', 'Floral', 'Sweet', 'Warm', 'Bakery','Spicy' ]
sagar_tasks = ['Intensity',
 'Pleasantness',
 'Fishy',
 'Burnt',
 'Sour',
 'Decayed',
 'Musky',
 'Fruity',
 'Sweaty',
 'Cool',
 'Floral',
 'Sweet',
 'Warm',
 'Bakery',
 'Spicy']

dravinak_descriptors = ['0.1',
 '1.1',
 '2.1',
 '3.1',
 '4.1',
 '5.1',
 '6.1',
 '7.1',
 '8.1',
 '9.1',
 '10.1',
 '11.1',
 '12.1',
 '13.1',
 '14.1',
 '15.1',
 '16.1',
 '17.1',
 '18.1',
 '19.1',
 '20.1',
 '21.1',
 '22.1',
 '23.1',
 '24.1',
 '25.1',
 '26.1',
 '27.1',
 '28.1',
 '29.1',
 '30.1',
 '31.1',
 '32.1',
 '33.1',
 '34.1',
 '35.1',
 '36.1',
 '37.1',
 '38.1',
 '39.1',
 '40.1',
 '41.1',
 '42.1',
 '43.1',
 '44.1',
 '45.1',
 '46.1',
 '47.1',
 '48.1',
 '49.1',
 '50.1',
 '51.1',
 '52.1',
 '53.1',
 '54.1',
 '55.1',
 '56.1',
 '57.1',
 '58.1',
 '59.1',
 '60.1',
 '61.1',
 '62.1',
 '63.1',
 '64.1',
 '65.1',
 '66.1',
 '67.1',
 '68.1',
 '69.1',
 '70.1',
 '71.1',
 '72.1',
 '73.1',
 '74.1',
 '75.1',
 '76.1',
 '77.1',
 '78.1',
 '79.1',
 '80.1',
 '81.1',
 '82.1',
 '83.1',
 '84.1',
 '85.1',
 '86.1',
 '87.1',
 '88.1',
 '89.1',
 '90.1',
 '91.1',
 '92.1',
 '93.1',
 '94.1',
 '95.1',
 '96.1',
 '97.1',
 '98.1',
 '99.1',
 '100.1',
 '101.1',
 '102.1',
 '103.1',
 '104.1',
 '105.1',
 '106.1',
 '107.1',
 '108.1',
 '109.1',
 '110.1',
 '111.1',
 '112.1',
 '113.1',
 '114.1',
 '115.1',
 '116.1',
 '117.1',
 '118.1',
 '119.1',
 '120.1',
 '121.1',
 '122.1',
 '123.1',
 '124.1',
 '125.1',
 '126.1',
 '127.1',
 '128.1',
 '129.1',
 '130.1',
 '131.1',
 '132.1',
 '133.1',
 '134.1',
 '135.1',
 '136.1',
 '137.1',
 '138.1',
 '139.1',
 '140.1',
 '141.1',
 '142.1',
 '143.1',
 '144.1',
 '145.1']

dravinsk_tasks = ['FRUITY,CITRUS',
 'LEMON',
 'GRAPEFRUIT',
 'ORANGE',
 'FRUITY,OTHER THAN CITRUS',
 'PINEAPPLE',
 'GRAPE JUICE',
 'STRAWBERRY',
 'APPLE, FRUIT',
 'PEAR',
 'CANTALOUPE, HONEYDEW MELON',
 'PEACH FRUIT',
 'BANANA',
 'FLORAL',
 'ROSE',
 'VIOLETS',
 'LAVENDER',
 'COLOGNE',
 'MUSK',
 'PERFUMERY',
 'FRAGRANT',
 'AROMATIC',
 'HONEY',
 'CHERRY, BERRY',
 'ALMOND',
 'NAIL POLISH REMOVER',
 'NUTTY, WALNUT ETC.',
 'SPICY',
 'CLOVE',
 'CINNAMON',
 'LAUREL LEAVES',
 'TEA LEAVES',
 'SEASONING FOR MEAT',
 'BLACK PEPPER',
 'GREEN PEPPER',
 'DILL',
 'CARAWAY',
 'OAK WOOD,COGNAC',
 'WOODY, RESINOUS',
 'CEDARWOOD',
 'MOTHBALLS',
 'MINTY, PEPPERMINT',
 'CAMPHOR',
 'EUCALYPTUS',
 'CHOCOLATE',
 'VANILLA',
 'SWEET',
 'MAPLE SYRUP',
 'CARAMEL',
 'MALTY',
 'RAISINS',
 'MOLASSES',
 'COCONUT',
 'ANISE, LICORICE',
 'ALCOHOLIC',
 'ETHERISH, ANAESTHETIC',
 'CLEANING FLUID',
 'GASOLINE, SOLVENT',
 'TURPENTINE, PINE OIL',
 'GERANIUM LEAVES',
 'CELERY',
 'FRESH GREEN VEGETABLES',
 'CRUSHED WEEDS',
 'CRUSHED GRASS',
 'HERBAL, GREEN,CUTGRASS',
 'RAW CUCUMBER',
 'HAY',
 'GRAINY (AS GRAIN)',
 'YEASTY',
 'BAKERY (FRESH BREAD)',
 'SOUR MILK',
 'FERMENTED, ROTTEN FRUIT',
 'BEERY',
 'SOAPY',
 'LEATHER',
 'CARDBOARD',
 'ROPE',
 'WET PAPER',
 'WET WOOL,WET DOG',
 'DIRTY LINEN',
 'STALE',
 'MUSTY, EARTHY, MOLDY',
 'RAW POTATO',
 'MOUSE',
 'MUSHROOM',
 'PEANUT BUTTER',
 'BEANY',
 'EGGY, FRESH EGGS',
 'BARK,BIRCHBARK',
 'CORK',
 'BURNT,SMOKY',
 'FRESH TOBACCO SMOKE',
 'INCENSE',
 'COFFEE',
 'STALE TOBACCO SMOKE',
 'BURNT PAPER',
 'BURNT MILK',
 'BURNT RUBBER',
 'TAR',
 'CREOSOTE',
 'DISINFECTANT, CARBOLIC',
 'MEDICINAL',
 'CHEMICAL',
 'BITTER',
 'SHARP, PUNGENT, ACID',
 'SOUR, VINEGAR',
 'SAUERKRAUT',
 'AMMONIA',
 'URINE',
 'CAT URINE',
 'FISHY',
 'KIPPERY,SMOKED FISH',
 'SEMINAL, SPERM-LIKE',
 'NEW RUBBER',
 'SOOTY',
 'BURNT CANDLE',
 'KEROSENE',
 'OILY, FATTY',
 'BUTTERY, FRESH BUTTER',
 'PAINT',
 'VARNISH',
 'POPCORN',
 'FRIED CHICKEN',
 'MEATY ( COOKED,GOOD)',
 'SOUPY',
 'COOKED VEGETABLES',
 'RANCID',
 'SWEATY',
 'CHEESY',
 'HOUSEHOLD GAS',
 'SULFIDIC',
 'GARLIC, ONION',
 'METALLIC',
 'BLOOD, RAW MEAT',
 'ANIMAL',
 'SEWER',
 'PUTRID, FOUL, DECAYED',
 'FECAL, LIKE MANURE',
 'CADAVEROUS',
 'SICKENING',
 'DRY, POWDERY',
 'CHALKY',
 'LIGHT',
 'HEAVY',
 'COOL,COOLING',
 'WARM']

