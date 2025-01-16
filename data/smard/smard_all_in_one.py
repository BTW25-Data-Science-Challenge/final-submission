import sys
import pandas as pd
import numpy as np
from pathlib import Path
import requests, datetime
from io import StringIO
#from datetime import datetime

#-------------translation for Balancing:------------------
balancing_id={
    #automatic frequency, tag=af
    "automatic_frequency":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume activated (+) [MWh] Calculated resolutions":"af_E_Volume_Activated_Plus_MWh",
        "Volume activated (-) [MWh] Calculated resolutions":"af_E_Volume_Activated_Minus_MWh",
        "Activation price (+) [€/MWh] Calculated resolutions":"af_Activation_Price_Plus_EUR/MWh",
        "Activation price (-) [€/MWh] Calculated resolutions":"af_Activation_Price_Minus_EUR/MWh",
        "Volume procured (+) [MW] Calculated resolutions":"af_E_Volume_Procured_Plus_MW",
        "Volume procured (-) [MW] Calculated resolutions":"af_E_Volume_Procured_Minus_MW",
        "Procurement price (+) [€/MW] Calculated resolutions":"af_Procurement_Price_Plus_EUR/MW",
        "Procurement price (-) [€/MW] Calculated resolutions":"af_Procurement_Price_Minus_EUR/MW",
    },
    #tag=mf
    "manual_frequency":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume activated (+) [MWh] Calculated resolutions":"mf_E_Volume_Activated_Plus_MWh",
        "Volume activated (-) [MWh] Calculated resolutions":"mf_E_Volume_Activated_Minus_MWh",
        "Activation price (+) [€/MWh] Calculated resolutions":"mf_Activation_Price_Plus_EUR/MWh",
        "Activation price (-) [€/MWh] Calculated resolutions":"mf_Activation_Price_Minus_EUR/MWh",
        "Volume procured (+) [MW] Calculated resolutions":"mf_E_Volume_Procured_Plus_MW",
        "Volume procured (-) [MW] Calculated resolutions":"mf_E_Volume_Procured_Minus_MW",
        "Procurement price (+) [€/MW] Calculated resolutions":"mf_Procurement_Price_Plus_EUR/MW",
        "Procurement price (-) [€/MW] Calculated resolutions":"mf_Procurement_Price_Minus_EUR/MW",
    },
     #balancing energy
    "balancing_energy":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume (+) [MWh] Calculated resolutions":"E_Volume_Calculated_Plus_MWh",
        "Volume (-) [MWh] Calculated resolutions":"E_Volume_Calculated_Minus_MWh",
        "Price [€/MWh] Calculated resolutions":"Price_Calculated_EUR/MWh",
        "Net income [€] Calculated resolutions":"Net_Income_EUR",
    },
    #costs
    "costs":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Balancing services [€] Calculated resolutions":"Balancing_Services_Calculated_EUR",
        "Network security [€] Calculated resolutions":"Network_Security_Calculated_EUR",
        "Countertrading [€] Calculated resolutions":"Countertrading_Calculated_EUR",
    },
    #frequency_containment_reserve
    "frequency_containment":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume procured [MW] Calculated resolutions":"E_Volume_Procured_Calculated_MW",
        "Procurement price [€/MW] Calculated resolutions":"Price_Procument_Calculated_EUR/MW"
    },
    "imported_balancing_services":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Austria [MWh] Calculated resolutions":"import_E_Austria_Calculated_MWh",
    },
    "exported_balancing_services":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Austria [MWh] Calculated resolutions":"export_E_Austria_Calculated_MWh",
    }         
}    

#actual consumption tag=actual
electricity_consumption_id={
    "actual":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Total (grid load) [MWh] Calculated resolutions":"actual_E_Total_Gridload_MWh",
        "Residual load [MWh] Calculated resolutions":"actual_E_Residual_Load_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"actual_E_Hydro_Pumped_Storage_MWh",
    },
    #forecasted consumption tag=forecast
    "forecast":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Total (grid load) [MWh] Calculated resolutions":"forecast_E_Total_Gridload_MWh",
        "Residual load [MWh] Calculated resolutions":"forecast_actual_E_Residual_Load_MWh"
    }
}

electricity_generation_id={
    #actual generation
    "actual":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MWh] Calculated resolutions":"actual_generation_E_Biomass_MWh",
        "Hydropower [MWh] Calculated resolutions":"actual_generation_E_Hydropower_MWh",
        "Wind offshore [MWh] Calculated resolutions":"actual_generation_E_Windoffshore_MWh",
        "Wind onshore [MWh] Calculated resolutions":"actual_generation_E_Windonshore_MWh",
        "Photovoltaics [MWh] Calculated resolutions":"actual_generation_E_Photovoltaics_MWh",
        "Other renewable [MWh] Calculated resolutions":"actual_generation_E_OtherRenewable_MWh",
        "Nuclear [MWh] Calculated resolutions":"actual_generation_E_Nuclear_MWh",
        "Lignite [MWh] Calculated resolutions":"actual_generation_E_Lignite_MWh",
        "Hard coal [MWh] Calculated resolutions":"actual_generation_E_HardCoal_MWh",
        "Fossil gas [MWh] Calculated resolutions":"actual_generation_E_FossilGas_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"actual_generation_E_HydroPumpedStorage_MWh",
        "Other conventional [MWh] Calculated resolutions":"actual_generation_E_OtherConventional_MWh"
    },
    
    #forecastet generation day ahead
    "forecast":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MWh] Calculated resolutions":"forecast_generation_E_Biomass_MWh",
        "Hydropower [MWh] Calculated resolutions":"forecast_generation_E_Hydropower_MWh",
        "Wind offshore [MWh] Calculated resolutions":"forecast_generation_E_Windoffshore_MWh",
        "Wind onshore [MWh] Calculated resolutions":"forecast_generation_E_Windonshore_MWh",
        "Photovoltaics [MWh] Calculated resolutions":"forecast_generation_E_Photovoltaics_MWh",
        "Other renewable [MWh] Calculated resolutions":"forecast_generation_E_OtherRenewable_MWh",
        "Nuclear [MWh] Calculated resolutions":"forecast_generation_E_Nuclear_MWh",
        "Lignite [MWh] Calculated resolutions":"forecast_generation_E_Lignite_MWh",
        "Hard coal [MWh] Calculated resolutions":"forecast_generation_E_HardCoal_MWh",
        "Fossil gas [MWh] Calculated resolutions":"forecast_generation_E_FossilGas_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"forecast_generation_E_HydroPumpedStorage_MWh",
        "Other [MWh] Calculated resolutions":"forecast_generation_E_Other_MWh",
        "Total [MWh] Original resolutions":"forecast_generation_E_Total_MWh",
        "Photovoltaics and wind [MWh] Calculated resolutions":"forecast_generation_E_PhotovoltaicsAndWind_MWh",
        "Other [MWh] Original resolutions":"forecast_generation_E_Original_MWh"
    },

    #installed generation capacity
    #key=instGenCapacity
    "installed_generation_capacity":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MW] Calculated resolutions":"instGenCapacity_E_Biomass_MW",
        "Hydropower [MW] Calculated resolutions":"instGenCapacity_E_Hydropower_MW",
        "Wind offshore [MW] Calculated resolutions":"instGenCapacity_E_Windoffshore_MW",
        "Wind onshore [MW] Calculated resolutions":"instGenCapacity_E_Windonshore_MW",
        "Photovoltaics [MW] Calculated resolutions":"instGenCapacity_E_Photovoltaics_MW",
        "Other renewable [MW] Calculated resolutions":"instGenCapacity_E_OtherRenewable_MW",
        "Nuclear [MW] Calculated resolutions":"instGenCapacity_E_Nuclear_MW",
        "Lignite [MW] Calculated resolutions":"instGenCapacity_E_Lignite_MW",
        "Hard coal [MW] Calculated resolutions":"instGenCapacity_E_HardCoal_MW",
        "Fossil gas [MW] Calculated resolutions":"instGenCapacity_E_FossilGas_MW",
        "Hydro pumped storage [MW] Calculated resolutions":"instGenCapacity_E_HydroPumpedStorage_MW",
        "Other conventional [MW] Calculated resolutions":"instGenCapacity_E_OtherConventional_MW"
    }
}

market_id={
    #key=dayAhead
    "day_ahead_prices":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Germany/Luxembourg [€/MWh] Original resolutions":"dayAhead_Price_GermanyAndLuxembourg_EUR/MWh",
        "∅ DE/LU neighbours [€/MWh] Original resolutions":"dayAhead_Price_GermanyAndLuxembourgAverage_EUR/MWh",
        "Belgium [€/MWh] Original resolutions":"dayAhead_Price_Belgium_EUR/MWh",
        "Denmark 1 [€/MWh] Original resolutions":"dayAhead_Price_Denmark1_EUR/MWh",
        "Denmark 2 [€/MWh] Original resolutions":"dayAhead_Price_Denmark2_EUR/MWh",
        "France [€/MWh] Original resolutions":"dayAhead_Price_France_EUR/MWh",
        "Netherlands [€/MWh] Original resolutions":"dayAhead_Price_Netherlands_EUR/MWh",
        "Norway 2 [€/MWh] Original resolutions":"dayAhead_Price_Norway2_EUR/MWh",
        "Austria [€/MWh] Original resolutions":"dayAhead_Price_Austria_EUR/MWh",
        "Poland [€/MWh] Original resolutions":"dayAhead_Price_Poland_EUR/MWh",
        "Sweden 4 [€/MWh] Original resolutions":"dayAhead_Price_Sweden4_EUR/MWh",
        "Switzerland [€/MWh] Original resolutions":"dayAhead_Price_Switzerland_EUR/MWh",
        "Czech Republic [€/MWh] Original resolutions":"dayAhead_Price_CzechRepublic_EUR/MWh",
        "DE/AT/LU [€/MWh] Original resolutions":"dayAhead_Price_DE/AT/LU_EUR/MWh",
        "Northern Italy [€/MWh] Original resolutions":"dayAhead_Price_NothernItaly_EUR/MWh",
        "Slovenia [€/MWh] Original resolutions":"dayAhead_Price_Slovenia_EUR/MWh",
        "Hungary [€/MWh] Original resolutions":"dayAhead_Price_Hungary_EUR/MWh"
    },
    
    "cross_border_physical":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Net export [MWh] Calculated resolutions":"E_NetExport_crossBorderPhysical_MWh",
        "Netherlands (export) [MWh] Calculated resolutions":"E_NetherlandExport_corssBorderPhysical_MWh",
        "Netherlands (import) [MWh] Calculated resolutions":"E_NetherlandImport_corssBorderPhysical_MW",
        "Switzerland (export) [MWh] Calculated resolutions":"E_SwitzerlandExport_corssBorderPhysical_MWh",
        "Switzerland (import) [MWh] Calculated resolutions":"E_SwitzerlandImport_corssBorderPhysical_MWh",
        "Denmark (export) [MWh] Calculated resolutions":"E_DenmarkExport_corssBorderPhysical_MWh",
        "Denmark (import) [MWh] Calculated resolutions":"E_Denmark_Import_corssBorderPhysical_MWh",
        "Czech Republic (export) [MWh] Calculated resolutions":"E_CzechrepublicExport_corssBorderPhysical_MWh",
        "Czech Republic (import) [MWh] Calculated resolutions":"E_CzechrepublicImport_corssBorderPhysical_MWh",
        "Luxembourg (export) [MWh] Calculated resolutions":"E_LuxembourgExport_corssBorderPhysical_MWh",
        "Luxembourg (import) [MWh] Calculated resolutions":"E_LuxembourgImport_corssBorderPhysical_MWh",
        "Sweden (export) [MWh] Calculated resolutions":"E_SwedenExport_corssBorderPhysical_MWh",
        "Sweden (import) [MWh] Calculated resolutions":"E_SwedenImportv_corssBorderPhysical_MWh",
        "Austria (export) [MWh] Calculated resolutions":"E_AustriaExport_corssBorderPhysical_MWh",
        "Austria (import) [MWh] Calculated resolutions":"E_AustriaImport_corssBorderPhysical_MWh",
        "France (export) [MWh] Calculated resolutions":"E_FranceExport_corssBorderPhysical_MWh",        
        "France (import) [MWh] Calculated resolutions":"E_FranceImport_corssBorderPhysical_MWh",
        "Poland (export) [MWh] Calculated resolutions":"E_PolandExport_corssBorderPhysical_MWh",
        "Poland (import) [MWh] Calculated resolutions":"E_PolandImport_corssBorderPhysical_MWh",
        "Norway (export) [MWh] Calculated resolutions":"E_NorwayExport_corssBorderPhysical_MWh",
        "Norway (import) [MWh] Calculated resolutions":"E_NorwayImport_corssBorderPhysical_MWh",
        "Belgium (export) [MWh] Calculated resolutions":"E_BelgiumExport_corssBorderPhysical_MWh",
        "Belgium (import) [MWh] Calculated resolutions":"E_BelgiumImport_corssBorderPhysical_MWh",
    },
    "scheudled_commercial_exchanges":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Net export [MWh] Calculated resolutions":"E_NetExport_MWh",
        "Netherlands (export) [MWh] Calculated resolutions":"E_NetherlandExport_MWh",
        "Netherlands (import) [MWh] Calculated resolutions":"E_NetherlandImport_MW",
        "Switzerland (export) [MWh] Calculated resolutions":"E_SwitzerlandExport_MWh",
        "Switzerland (import) [MWh] Calculated resolutions":"E_SwitzerlandImport_MWh",
        "Denmark (export) [MWh] Calculated resolutions":"E_DenmarkExport_MWh",
        "Denmark (import) [MWh] Calculated resolutions":"E_Denmark_Import_MWh",
        "Czech Republic (export) [MWh] Calculated resolutions":"E_CzechrepublicExport_MWh",
        "Czech Republic (import) [MWh] Calculated resolutions":"E_CzechrepublicImport_MWh",
        "Luxembourg (export) [MWh] Calculated resolutions":"E_LuxembourgExport_MWh",
        "Luxembourg (import) [MWh] Calculated resolutions":"E_LuxembourgImport_MWh",
        "Sweden (export) [MWh] Calculated resolutions":"E_SwedenExport_MWh",
        "Sweden (import) [MWh] Calculated resolutions":"E_SwedenImport_MWh",
        "Austria (export) [MWh] Calculated resolutions":"E_AustriaExport_MWh",
        "Austria (import) [MWh] Calculated resolutions":"E_AustriaImport_MWh",
        "France (export) [MWh] Calculated resolutions":"E_FranceExport_MWh",        
        "France (import) [MWh] Calculated resolutions":"E_FranceImport_MWh",
        "Poland (export) [MWh] Calculated resolutions":"E_PolandExport_MWh",
        "Poland (import) [MWh] Calculated resolutions":"E_PolandImport_MWh",
        "Norway (export) [MWh] Calculated resolutions":"E_NorwayExport_MWh",
        "Norway (import) [MWh] Calculated resolutions":"E_NorwayImport_MWh",
        "Belgium (export) [MWh] Calculated resolutions":"E_BelgiumExport_MWh",
        "Belgium (import) [MWh] Calculated resolutions":"E_BelgiumImport_MWh",
    }
}

def main():

    output_path = sys.argv[1]

    dict_ids = [balancing_id["automatic_frequency"],
                balancing_id["balancing_energy"],
                balancing_id["costs"],
                balancing_id["exported_balancing_services"],
                balancing_id["frequency_containment"],
                balancing_id["imported_balancing_services"],
                balancing_id["manual_frequency"],
                electricity_consumption_id["actual"],
                electricity_consumption_id["forecast"],
                electricity_generation_id["actual"],
                electricity_generation_id["forecast"],
                market_id["cross_border_physical"],
                market_id["scheudled_commercial_exchanges"],
                market_id["day_ahead_prices"]    
    ]
    
    final_df = None

    for i in range(13):
        working_df = download(i)
        working_df = new_format(working_df, dict_ids[i])

        if i > 0:
            working_df=working_df.drop(working_df.columns[1],axis=1)
        #only called once
        if final_df is None:
            final_df = working_df
        else:
            final_df = pd.merge(final_df, working_df, on=working_df.columns[0], how='inner', copy=True)
    
    final_df.to_csv(output_path, sep=',', index=False)

    #use gzip to compress .csv outputfile to <file_out>.gz
    path_object = Path(output_path)
    output_pathgz = path_object.with_suffix('.gz')
    final_df.to_csv(output_pathgz, sep=',', index=False, compression='gzip')




def download(download_id):
    #14 different files
    match download_id:
        # AUTOMATIC FREQUENCY RESTORATION
        case 0:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[18004368,18004369,18004370,18004351,18004371,18004372,18004373,18004374],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # BALANCING ENERGY
        case 1:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[15004383,15004384,15004382,15004390],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')        
        # COSTS  
        case 2:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[16004391,16000419,16000418],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # EXPORTED BALANCING SERVICES
        case 3:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[20004385],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        #FREQUENCY CONTAINMENT RESERVE
        case 4:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[17004363, 17004367],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # IMPORTED BALANCING SERVICES
        case 5:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[21004386],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # MANUAL FREQUENCY RESTORATION RESERVE
        case 6:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[19004377,19004375,19004376,19004352,19004378,19004379,19004380,19004381],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        
        #electricity consumption, actual
        case 7:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[5000410,5004387,5004359],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        #forecast consumption
        case 8:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[6000411,6004362],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        #electricity generation actual
        case 9:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[1001224,1004066,1004067,1004068,1001223,1004069,1004071,1004070,1001226,1001228,1001227,1001225],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        #electricity generation forecast
        case 10:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[2000122,2005097,2000715,2003791,2000123,2000125],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        #MARKET
        # CROSSBORDER FLOWS
        case 11:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[31004963,31004736,31004737,31004740,31004741,31004988,31004990,31004992,31004994,31004738,31004742,31004743,31004744,31004880,31004881,31004882,31004883,31004884,31004885,31004886,31004887,31004888,31004739],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # CROSSBORDER SCHEDULED FLOWS
        case 12:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[22004629,22004722,22004724,22004404,22004409,22004545,22004546,22004548,22004550,22004551,22004552,22004405,22004547,22004403,22004406,22004407,22004408,22004410,22004412,22004549,22004553,22004998,22004712],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        # DAYAHEAD
        case 13:
            response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                data='{"request_form":[{"format":"CSV","moduleIds":[8004169,8004170,8000251,8005078,8000252,8000253,8000254,8000255,8000256,8000257,8000258,8000259,8000260,8000261,8000262,8004996,8004997],"region":"DE","timestamp_from":1420066800000,"timestamp_to":'+str(int(datetime.datetime.today().timestamp()))+'000,"type":"discrete","language":"en","resolution":"hour"}]}')
        
    csvfile_data = response.content.decode('utf-8-sig')
    download_df = pd.read_csv(StringIO(csvfile_data), sep=";", header=[0], na_values='-', low_memory=False)
    
    return download_df



def new_format(df, my_dict):
        
    #use fitting dict to rename table head
    df.rename(columns=my_dict, inplace=True)
    
    #change Datetime_format; replace '-' with np.nan
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df['End_Date'] = pd.to_datetime(df['End_Date'])
    df.replace("-",np.nan, inplace=True)

    #remove , seperator for thousand
    df.replace(",","", inplace=True, regex=True)
    
    return df
    

def merge(fin_df, work_df, i):

    if i > 0:
        work_df=work_df.drop(work_df.columns[1],axis=1)
    
    fin_df = pd.merge(fin_df, work_df, on=work_df.columns[0], how='inner', copy=True)
    


if __name__ == '__main__':
    main()