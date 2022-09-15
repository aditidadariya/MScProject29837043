#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: Config_1.py

This file contains the configurations like variable declaration and variable assignment with their values

"""

############################# VARIABLE DECLARATION ############################

# location_of_file variable is defined to store the dataset file location
location_of_file = '/Users/aditidadariya/Aditi Personal/UK Universities/University of Reading/Modules/CSMPR21_MSc Project/Project/'  #'bank.xlsx'
originalfile = "bank.xlsx"
cleanfile = "bank_cleandata.xlsx"
cluster0data = "Data_Cluster0.xlsx"
cluster1data = "Data_Cluster1.xlsx"
cluster2data = "Data_Cluster2.xlsx"
cluster3data = "Data_Cluster3.xlsx"
cluster4data = "Data_Cluster4.xlsx"

# Define a list of all categories
categorylist = ['2DFINE', 'AAB ENTERPRISES', 'ACCOUNT', 'ACQ', 'ADJ', 'AEPS', 'AIRPAY', 'AIRTEL', 'ALT DIGITAL', 'APPNIT', 'ATLAS', 'ATM', 'AVENUES INDIA PRIVATE LIM', 'AWARDS', 'AXEL',
                'BAJAJ ALLIANZ', 'Bank Guar', 'BASIST', 'BBPS', 'BEAT CSH', 'BEST', 'BHAGALPUR ELECTRICITY DIS', 'BIDDERBOY CORPORATION', 'BIGTREE ENTERTAINMENT PRI', 'BILLPAY', 'BIRLA', 'BLUE DART', 'BONUSHUB', 'BOOK', 'BOWMAN', 'BSES RAJDHANI POWER LIMIT', 'BSES YAMUNA POWER LIMITED', 
                'CALLHEALTH ONLINE SERVICE', 'CASH', 'CATENA TECHNOLOGIES PRIVA', 'CB', 'CDUTWL', 'CESS', 'Charges', 'CHQ', 'CLEARCAR RENTAL PRIVATE L', 'Closure', 'COMMDEL CONSUTING SERVICE', 'CONFEDERATION OF INDIAN I', 'CREDIT', 'CRY', 'CULLIGENCE SOFTWARE', 
                'DD', 'DEFERRED', 'DEN NETWORKS LIMITED', 'DHL', 'DIGITAL', 'DISH INFRA SERVICES PRIVA', 'Dr.', 'Draw Down', 'dsb',
                'E SUVIDHA', 'E-BILLING SOLUTIONS PRIVA', 'EIH', 'ELEGANCE FASHION AND YOU', 'EMBEE SOFTWARE PVT LTD', 'ETRAVELVALUE', 
                'FACEBOOK', 'FAKE', 'FD', 'fdrl', 'FEDEX', 'FEE', 'FERNS', 'FINSPIRE SOLUTIONS PRIVAT', 'FIXED MOBILE', 'FROM RBL BANK LTD', 'FT ZL', 'FTR', 'FUJIAN', 'FUTURE RETAIL LIMITED', 'FX CONV CHGS', 'GOLD', 'GOLF', 'GST',
                'HAPPYDEAL18', 'hdfc', 'HINDUSTAN SERVICE STATION', 'HOSPITAL', 'HUDA', 
                'IFSC', 'IMAR', 'imps', 'Indfor', 'Indiaforensic', 'INDIAIDEAS', 'INDIAN OIL', 'INDO', 'INFOEDGE INDIA LTD', 'INT', 'INWARD', 'IO', 'IRCTC', 'IRTT', 'IW CLG', 'IWSTHCTS1', 
                'JANA', 'JUST DIAL LIMITED','KAYGEES', 'KRYPTOS MOBILE PRIVATE LI', 'LINKEDIN SINGAPORE PTE LT', 'LOCALCUBE COMMERCE PRIVAT', 
                'MAD', 'MAESTRO', 'MARGIN', 'MARKING', 'MASTER', 'MAW', 'MDR', 'MEHAR ENTERTAINMENT PRIVA', 'MEHER ENTERTAINMENT PRIVA', 'MERU CAB COMPANY PRIVATE', 'METRO INFRASYS PRIVATE LI', 'MICRO', 'MS EMBEE SOFTWARE', 'MTNL', 'MUZAFFARPUR VIDYUT VITARA', 
                'NATIONAL', 'neft', 'NETHOPE', 'NFS', 'NOMISMA MOBILE SOLUTIONS', 'NORTH DELHI POWER LTD', 'NoTransDetail', 'npc',
                'OBEROI', 'OFT', 'OM GANESH', 'OMNICOM MEDIA GROUP INDIA', 'ONLINE', 'ORAVEL STAYS PRIVATE LIMI', 'ORTT', 'OUTWARD', 'OW RET', 'oxygen', 
                'PASFAR TECHNOLOGIES PRIVA', 'PAYGATE INDIA PRIVATE LIM', 'Payments', 'Payoff', 'PENALITY', 'PENALTY', 'PENDING', 'PENLTY', 'PVR LIMITED',  
                'QUINSEL SERVICES PRIVATE', 'QWIKCILVER SOLUTIONS', 'RAI', 'RAJMATAJI', 'RECOVERY', 'REJ', 'REP', 'Repayment', 'REV', 'REVERSE LOGISTICS COMPANY', 'RF', 'RMCPL', 'ROYAL INFOSYS', 'RPT', 'RTGS', 'RUPAY',
                'sbi', 'SEND MONEY', 'SERVICE TX', 'SHENZHEN JUSTTIDE TECH CO', 'SHREEJEE PRIME INFOTECH P', 'SKORYDOV', 'SMARTSERV INFOSERVICES PR', 'SND LIMITED', 'SONATA FINANCE PRIVATE LI', 'SPORTS', 'SRS LIMITED', 'SRVC TX', 'STAMP', 'SWACHH', 'SYNERGISTIC FINANCIAL NET',
                'TAKEOVER', 'TATA', 'TAX', 'TDS', 'TELECOM', 'TELEPOWER', 'TELPOWER', 'THE UNIWORLD CITY', 'TITAN', 'TNT', 'TO For', 'TPF', 'TRANGLO', 'TransNum', 'TRAVEL', 'TRF', 
                'UPI', 'USD', 'VAISHANVI', 'VENDOR', 'VINILOK SOLUTIONS', 'VISA', 'VODAFONE', 'VOUCHER', 'VOYLLA FASHIONS PRIVATE L', 'WAIVE', 'WORLD WIDE FUND', 'ZAAK EPAYMENT SERVICES PR', 'ZEN', 'RefNum'] # 'Self'

businesslist = ['BIDDERBOY CORPORATION','CONFEDERATION OF INDIAN I', 'DEN NETWORKS LIMITED', 'INFOEDGE INDIA LTD', 'LOCALCUBE COMMERCE PRIVAT','MEHAR ENTERTAINMENT PRIVA', 'METRO INFRASYS PRIVATE LI', 
                'ORAVEL STAYS PRIVATE LIMI', 'PAYGATE INDIA PRIVATE LIM', 'SHREEJEE PRIME INFOTECH P', 'SONATA FINANCE PRIVATE LI', 'SYNERGISTIC FINANCIAL NET', 'VOYLLA FASHIONS PRIVATE L', 
                'AAB ENTERPRISES', 'BIGTREE ENTERTAINMENT PRI', 'E-BILLING SOLUTIONS PRIVA', 'FROM RBL BANK LTD', 'MEHER ENTERTAINMENT PRIVA', 'QUINSEL SERVICES PRIVATE', 'BSES RAJDHANI POWER LIMIT', 
                'CATENA TECHNOLOGIES PRIVA', 'CLEARCAR RENTAL PRIVATE L', 'KRYPTOS MOBILE PRIVATE LI',  'MS EMBEE SOFTWARE', 'QWIKCILVER SOLUTIONS', 'SMARTSERV INFOSERVICES PR', 'SRS LIMITED', 
                'AVENUES INDIA PRIVATE LIM', 'BSES YAMUNA POWER LIMITED', 'CULLIGENCE SOFTWARE', 'DISH INFRA SERVICES PRIVA', 'ELEGANCE FASHION AND YOU', 'SND LIMITED', 'ZAAK EPAYMENT SERVICES PR',
                'COMMDEL CONSUTING SERVICE', 'EMBEE SOFTWARE PVT LTD', 'MUZAFFARPUR VIDYUT VITARA', 'NOMISMA MOBILE SOLUTIONS', 'REVERSE LOGISTICS COMPANY', 'VINILOK SOLUTIONS', 'FUTURE RETAIL LIMITED',
                'NORTH DELHI POWER LTD', 'OMNICOM MEDIA GROUP INDIA', 'PASFAR TECHNOLOGIES PRIVA', 'SHENZHEN JUSTTIDE TECH CO', 'ALT DIGITAL', 'FINSPIRE SOLUTIONS PRIVAT', 'ROYAL INFOSYS', 'TATA']

courierservicelist = ['DHL', 'BLUE DART', 'FEDEX']

investmentlist = ['BAJAJ ALLIANZ','E SUVIDHA', 'BIRLA','FD']

mobileservicelist = ['FIXED MOBILE', 'MTNL', 'TELECOM', 'VODAFONE', 'AIRTEL', 'AIRPAY']

transfermodelist = ['RTGS', 'RUPAY', 'UPI', 'CASH', 'CHQ', 'CREDIT', 'IFSC', 'neft', 'Payments', 'ATM', 'INWARD', 'MAESTRO', 'Payoff', 'imps', 'NFS', 'REV', 'DD', 'fdrl', 'VISA', 'USD', 'BILLPAY', 
                    'Repayment', 'GST', 'SEND MONEY', 'TDS', 'PENDING']

chargeslist = ['Charges', 'FX CONV CHGS', 'PENALTY', 'PENALITY']

entertaintravellist = ['ETRAVELVALUE', 'PVR LIMITED', 'SPORTS', 'DIGITAL', 'GOLF', 'MERU CAB COMPANY PRIVATE', 'LINKEDIN SINGAPORE PTE LT', 'HAPPYDEAL18', 'IRCTC', 'SWACHH', 'INDIAIDEAS', 
                       'TRAVEL', 'VOUCHER']

taxlist = ['SRVC TX', 'TAX', 'SERVICE TX', 'PENLTY']

# special_char variable is defined to store any special characters
specialchar = "'"
# Define outliers list to store the outliers in data
outliers=[]
# Define models list to store the model object
models = []
# Define names list to store the name of models
names = []
# Define results list to store the accuracy score
results = []
# Define basicscore list to store the name and accuracy score
basic_score = []
# Define basicscore list to store the name and accuracy score
score = []
# Define finalresultslist to store cross validation score
final_results = []
# Define scoreslist list to store the depp learning scores
scores_list = list()
# Create list for GMM model
sscores=[]
sresults = {}
# define generator parameters
n_input = 10
n_features = 1
# list is defined to store the root mean squared errors
rmselist = list()
rmsetunelist = list()
# list are defined to hold 
testpredictions = list()
futurepredictions = list()
best_epochs = list()
# Defined a randstate variable to store the input for random_state in train_test_split function later
rand_state = [1,3,5,7]
# Define a eps variable to store different epochs values
eps = [10,15,20,25,50,100]
# evaluate parameters for ARIMA and VARMAX models
p_values = [1, 2, 3, 4, 5]
d_values = 0
q_values = range(0, 6)

