# This Repo
All data goes here, split into respective domains, as well as data priorities and inter-group logs.

# Data format
- data generation script & maybe example data
- story of data (origin, aquisition, etc...)
- output format
  - panda dataframes (csv)
  - separator: ,
- frequency
  - HOURLY
- missing values:
  - numpy.nan
  - keine Imputation (macht xAI)
- time window:
  - 01.01.15 - today
- timestamps:
  - pandas.Timestamp format (timezone: UTC)
  - ISO format: YYYY-MM-DD HH:MM:SS+HH:MM
- Energie:
  - float
- day, month, year, holiday:
  - int
- non existent values:
  - numpy.nan
- boolean values:
  - int
- names:
  - One Hot Encoding

# Naming conventions:
- Pysical Units:
  
        [Formelzeichen]_[Kürzel]_[actual/forecast]_[Einheit]

example: 
- T_temperature_forecast_C
- E_nuclear_actual_MWh
- E_Solar_forecast_MWh

Non Physical Units:

	[kürzel]_[Einheit]

example:
- holiday_bool
- day_ahead_prices_EURO

# Metrics
- MSE (how good for peaks), MAPE (general model performance)

# Train/Val/Test
- Val: 01.12.23 - 31.05.24
- Test: 01.05.24 - 30.11.24
- Train: Alles vorher

# Minimal Feature Set
- day-ahead-price
- wetter
- installed generation capacity
- ölpreis
- ...

# Data Priority list (WIP)
- smard - strompreise und derivate (abweichungen [n/a], moving average [hat Ansgar aus xAi gemacht], vorhersag-diskrepanzen etc...) (Moritz)
- wochentage, feiertage, ferien (in AutoGluon enthalten)
- [popular next day prediction](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Energiemarkt_entsoe) (Jordan) 
- [Wetter, unwetter](https://github.com/BTW25-Data-Science-Challenge/Data/blob/main/Data/Weather) (Felix)
- [Gas/Öl/Kohle Preise](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/OilGasCoal) (Clara)
- [geplante wartung von kraftwerken](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Energiemarkt_entsoe) (Jordan) 
- [cross-border energy & ressource flows (kosten, quantitäten)](https://github.com/BTW25-Data-Science-Challenge/Data/blob/main/Data/Smard/Market/generate_dataframe.ipynb) (Moritz - Teil von Smard)
- [carbon prize entwicklung, steuern , subventionen, regulierungen, embargos](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Carbon_price) (Malte)
- [corona lockdowns](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Corona-Pandemie) (Moritz)
- [gdp](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/GDP) (Clara)
- aktienindizes (energieunternehmen priorisiert) (Clara)
- [major social events (tourism, hotelauslastung)](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Major_social_events) (Malte)
- [inflation](https://github.com/BTW25-Data-Science-Challenge/Data/tree/main/Data/Inflation) (Clara)
- news embeddings (ungeplante outages)

# Sources etc.
https://github.com/BTW25-Data-Science-Challenge/AutoGluon/blob/main/Data_brainstorm.ipynb

# Ambassadery 
Intra group meeting logs

# HPC
[Siehe hier](https://compendium.hpc.tu-dresden.de/)
1) ssh Verbindung (ssh login2.alpha.hpc.tu-dresden.de)
2) workspace anlegen (ws_allocate -F horse --name="timeseries-forecast" --duration=100 --reminder=7 --mailaddress="unimail")
3) partition anfordern - interactiv oder batch (interactiv: srun -c 8 --mem=64G -t 500 -N 1 --gres=gpu:1 -p alpha --pty bash -l) (batch: batch job_file.sh)
4) module laden (e.g. ml GCCcore/11.3.0)
5) rechnen