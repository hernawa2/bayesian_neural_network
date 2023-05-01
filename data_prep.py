#airquality
import pandas as pd, os, numpy as np
path=r'C:\Users\hernawa2\stat618data\airqualitymadrid\csvs_per_year\csvs_per_year';
df_air=pd.read_csv(os.path.join(path,os.listdir(path)[0]));
station=df_air.station.drop_duplicates();
df_air[['date_tmy','hour']]=df_air['date'].str.split(' ',expand=True);
time_day=df_air.date_tmy.drop_duplicates().reset_index(drop=True);
time_day=list(np.sort(time_day.to_numpy()));
avg_no2=[];
avg_so2=[];
avg_co=[];
avgpm10=[];
avg_o3=[];
avg_ben=[];
avg_ebe=[];
avg_mxy=[];
avg_pxy=[];
avg_oxy=[];
avg_tch=[];
for j in range(len(time_day)):
    avg_no2.append(df_air[df_air.date_tmy==time_day[j]].NO_2.mean())
    avg_so2.append(df_air[df_air.date_tmy==time_day[j]].SO_2.mean())
    avg_co.append(df_air[df_air.date_tmy==time_day[j]].CO.mean())
    avgpm10.append(df_air[df_air.date_tmy==time_day[j]].PM10.mean())
    avg_o3.append(df_air[df_air.date_tmy==time_day[j]].O_3.mean())
    avg_ben.append(df_air[df_air.date_tmy==time_day[j]].BEN.mean())
    avg_ebe.append(df_air[df_air.date_tmy==time_day[j]].EBE.mean())
    avg_mxy.append(df_air[df_air.date_tmy==time_day[j]].MXY.mean())
    avg_pxy.append(df_air[df_air.date_tmy==time_day[j]].PXY.mean())
    avg_oxy.append(df_air[df_air.date_tmy==time_day[j]].OXY.mean())
    avg_tch.append(df_air[df_air.date_tmy==time_day[j]].TCH.mean())
avg_no2_tot=[];
avg_so2_tot=[];
avg_co_tot=[];
avg_pm10_tot=[];
avg_o3_tot=[];
avg_ben_tot=[];
avg_ebe_tot=[];
# avg_mxy_tot=[];
# avg_pxy_tot=[];
# avg_oxy_tot=[];
avg_tch_tot=[];
time=[];
for i in range(len(os.listdir(path))):
    temp=pd.read_csv(os.path.join(path,os.listdir(path)[i]));
    temp[['date_tmy','hour']]=temp['date'].str.split(' ',expand=True);
    time_day=temp.date_tmy.drop_duplicates().reset_index(drop=True);
    time_day=list(np.sort(time_day.to_numpy()));
    time.append(time_day);
    avg_no2_tot.append(temp[['date_tmy','NO_2']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_so2_tot.append(temp[['date_tmy','SO_2']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_co_tot.append(temp[['date_tmy','CO']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_pm10_tot.append(temp[['date_tmy','PM10']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_o3_tot.append(temp[['date_tmy','O_3']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_ben_tot.append(temp[['date_tmy','BEN']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_ebe_tot.append(temp[['date_tmy','EBE']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
#     avg_mxy_tot.append(temp[['date_tmy','MXY']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
#     avg_pxy_tot.append(temp[['date_tmy','PXY']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
#     avg_oxy_tot.append(temp[['date_tmy','OXY']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
    avg_tch_tot.append(temp[['date_tmy','TCH']].groupby('date_tmy').mean().reset_index(drop=True).iloc[:,0].to_list());
time_flat=[item for sublist in time for item in sublist];
avg_no2_flat=[item for sublist in avg_no2_tot for item in sublist];
avg_so2_flat=[item for sublist in avg_so2_tot for item in sublist];
avg_co_flat=[item for sublist in avg_co_tot for item in sublist];
avg_pm10_flat=[item for sublist in avg_pm10_tot for item in sublist];
avg_o3_flat=[item for sublist in avg_o3_tot for item in sublist];
avg_ben_flat=[item for sublist in avg_ben_tot for item in sublist];
avg_ebe_flat=[item for sublist in avg_ebe_tot for item in sublist];
avg_tch_flat=[item for sublist in avg_tch_tot for item in sublist];
df_name=pd.DataFrame({'date':time_flat, 'Nitrogen Diox.':avg_no2_flat, 'Sulphur Diox.':avg_so2_flat,
                   'Carbon Monox.':avg_co_flat, 'PM10':avg_pm10_flat, 'Ozone':avg_o3_flat,
                   'Benzine':avg_ben_flat, 'EthylBen.':avg_ebe_flat, 'Hydrocarb.':avg_tch_flat})
