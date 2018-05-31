# 


# implement poisson likelihood
- How to do the expectation computation
- Do laplace approximation
- 

# Numerical estimation of other likelihood?
- For both poisson prediction and if we want to have closed form
for Di and di, then we need to do that ugly formula to approximate
the distribution over f_ijk
- 


/home/duc/Documents/Research/SSVI-TF/venv/bin/python /home/duc/Documents/Research/SSVI-TF/Test/PTF_GME_3D.py
Generating synthetic data ... 
Generating synthetic data took:  3.1377904415130615
iteration:  100  - test error:  0.3592036740818719  - train error:  0.3529564098684713  - time:  1.986994743347168
iteration:  200  - test error:  0.2570735635197056  - train error:  0.2515002612809335  - time:  6.597862243652344


/home/duc/Documents/Research/SSVI-TF/venv/bin/python /home/duc/Documents/Research/SSVI-TF/Test/PTF_GME_3D.py
Generating synthetic data ... 
Generating synthetic data took:  2.961392879486084
iteration:  100  - test error:  0.3592036740818719  - train error:  0.3529564098684713  - time:  1.9860804080963135
iteration:  200  - test error:  0.2570735635197056  - train error:  0.2515002612809335  - time:  6.533876419067383



/home/duc/Documents/Research/SSVI-TF/venv/bin/python /home/duc/Documents/Research/SSVI-TF/Test/PTF_GME_3D.py
Generating synthetic data ... 
Generating synthetic data took:  3.032604932785034
iteration:  100  - test error:  0.3592036740818719  - train error:  0.3529564098684713  - time:  2.0360331535339355
iteration:  200  - test error:  0.2570735635197056  - train error:  0.2515002612809335  - time:  6.650970697402954
iteration:  300  - test error:  0.21022865110710687  - train error:  0.20570002673255244  - time:  11.633358001708984
iteration:  400  - test error:  0.17840529458256044  - train error:  0.17454076617228154  - time:  16.568132638931274
iteration:  500  - test error:  0.1510029903927793  - train error:  0.1479758350598796  - time:  21.307124853134155
iteration:  600  - test error:  0.13127777420272185  - train error:  0.12912221286751502  - time:  26.064617395401
iteration:  700  - test error:  0.11647201912470369  - train error:  0.11480115462100457  - time:  30.847421884536743
iteration:  800  - test error:  0.10505292649292626  - train error:  0.10303810667131136  - time:  35.633817195892334
iteration:  900  - test error:  0.09616633231340664  - train error:  0.09415046185626727  - time:  40.47223377227783
iteration:  1000  - test error:  0.08666253469876666  - train error:  0.08521401984000834  - time:  45.31665921211243
iteration:  1100  - test error:  0.08017199560186478  - train error:  0.07866150286968276  - time:  50.175862073898315
iteration:  1200  - test error:  0.07404183068518581  - train error:  0.0726858728551863  - time:  55.02214956283569
iteration:  1300  - test error:  0.06973946581130726  - train error:  0.06870084460811642  - time:  59.8567316532135
iteration:  1400  - test error:  0.06619231068699163  - train error:  0.06506266352976824  - time:  64.76375961303711
iteration:  1500  - test error:  0.06343673106561629  - train error:  0.0622839746238026  - time:  69.66635823249817
iteration:  1600  - test error:  0.060745204538311795  - train error:  0.05962180170522431  - time:  74.5755295753479
iteration:  1700  - test error:  0.05870198275158397  - train error:  0.0575283953480715  - time:  79.48594546318054
iteration:  1800  - test error:  0.05625655936189014  - train error:  0.05524447230028214  - time:  84.40996599197388
iteration:  1900  - test error:  0.05488187396961556  - train error:  0.05404327143334974  - time:  89.35840082168579
iteration:  2000  - test error:  0.053957294893893525  - train error:  0.05310411957213744  - time:  94.2691957950592
iteration:  2100  - test error:  0.05224377636930159  - train error:  0.051331477943352834  - time:  99.17630290985107
iteration:  2200  - test error:  0.05117842276252989  - train error:  0.0505370691465861  - time:  104.08762097358704
iteration:  2300  - test error:  0.050409435677412774  - train error:  0.04969749018600159  - time:  109.00669384002686
iteration:  2400  - test error:  0.04948896290736551  - train error:  0.04903241683165002  - time:  113.94161009788513
iteration:  2500  - test error:  0.049020793505967324  - train error:  0.04850018403127132  - time:  118.89849042892456
iteration:  2600  - test error:  0.04848851227544009  - train error:  0.047995963687536884  - time:  123.82733654975891
iteration:  2700  - test error:  0.04771062809404275  - train error:  0.04727563821876398  - time:  128.76792073249817
iteration:  2800  - test error:  0.04758870087792013  - train error:  0.0469394015271271  - time:  133.72612595558167
iteration:  2900  - test error:  0.04694875264662928  - train error:  0.04647467624949226  - time:  138.69473242759705
iteration:  3000  - test error:  0.046947546509293926  - train error:  0.04649269761393006  - time:  143.64437460899353
iteration:  3100  - test error:  0.04652555630345528  - train error:  0.04611127289679257  - time:  148.61247181892395
iteration:  3200  - test error:  0.04667697832295511  - train error:  0.04609606583061797  - time:  153.56885933876038
iteration:  3300  - test error:  0.046651756337950594  - train error:  0.04609044258164301  - time:  158.53720927238464
iteration:  3400  - test error:  0.04624017565783896  - train error:  0.045658818006432414  - time:  163.48303055763245
iteration:  3500  - test error:  0.04576596774786949  - train error:  0.04527904408122267  - time:  168.44366693496704
iteration:  3600  - test error:  0.045916249138741524  - train error:  0.04544246227070539  - time:  173.40922951698303
iteration:  3700  - test error:  0.046043171978227604  - train error:  0.04574867579366717  - time:  178.36292910575867
iteration:  3800  - test error:  0.045985265332019735  - train error:  0.04542713832998727  - time:  183.3557994365692
iteration:  3900  - test error:  0.04533141978751073  - train error:  0.04490320991810646  - time:  188.39495253562927
iteration:  4000  - test error:  0.045399175914891  - train error:  0.04507829257152091  - time:  193.41389727592468
iteration:  4100  - test error:  0.04556380875753743  - train error:  0.045087609779799045  - time:  198.43268513679504
iteration:  4200  - test error:  0.045500705743167204  - train error:  0.04514903673809493  - time:  203.4185938835144
iteration:  4300  - test error:  0.04560832398609491  - train error:  0.045332336481993425  - time:  208.424546957016
iteration:  4400  - test error:  0.04572937399614867  - train error:  0.045310714870940384  - time:  213.41501569747925
iteration:  4500  - test error:  0.045932042142971696  - train error:  0.04551508106822697  - time:  218.40621495246887
iteration:  4600  - test error:  0.04592561343926293  - train error:  0.04547753126768981  - time:  223.38215112686157
iteration:  4700  - test error:  0.04546398503442133  - train error:  0.044947343738795366  - time:  228.3814775943756
iteration:  4800  - test error:  0.04581170007138309  - train error:  0.04530762255695515  - time:  233.38234400749207
iteration:  4900  - test error:  0.04552940128147697  - train error:  0.045196946441190676  - time:  238.42318177223206
iteration:  5000  - test error:  0.045666851131815726  - train error:  0.045096306791191726  - time:  243.39247632026672
iteration:  5100  - test error:  0.045503452681053144  - train error:  0.044989747563541925  - time:  248.35698246955872
iteration:  5200  - test error:  0.04561518277023468  - train error:  0.045138685564773365  - time:  253.33540439605713
iteration:  5300  - test error:  0.046096556261640886  - train error:  0.04552035964286811  - time:  258.3301968574524
iteration:  5400  - test error:  0.045607453922786666  - train error:  0.04507067231597559  - time:  263.3147008419037
iteration:  5500  - test error:  0.04535174868871047  - train error:  0.04480263769171533  - time:  268.28724002838135
iteration:  5600  - test error:  0.04547156788359396  - train error:  0.04491685688943608  - time:  273.3160479068756
iteration:  5700  - test error:  0.04548709350872534  - train error:  0.044976563055942324  - time:  278.2964241504669
iteration:  5800  - test error:  0.0456256235247742  - train error:  0.04513346343216557  - time:  283.26934123039246
iteration:  5900  - test error:  0.045258741681115454  - train error:  0.04482566776523693  - time:  288.2773070335388


