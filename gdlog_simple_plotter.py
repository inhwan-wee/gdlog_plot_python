#!/usr/bin/env python3

import sys
import signal
import base64
import datetime
import io
import struct
import csv

import numpy as np

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def signal_handler(signal, frame):
    print('\npressed ctrl + c!!!\n')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                prevent_initial_callbacks=True)

df = pd.DataFrame()
df_pc = pd.DataFrame()
fcMcMode_index = []
fcMcMode_value = []
fcMcMode_color = []

prev_mission_clicks = 0
prev_gps_clicks = 0
prev_rpd_roll_clicks = 0
prev_rpd_pitch_clicks = 0
prev_rpd_down_clicks = 0
prev_yaw_clicks = 0
prev_vel_x_clicks = 0
prev_vel_y_clicks = 0
prev_vel_z_clicks = 0
prev_pos_n_clicks = 0
prev_pos_e_clicks = 0
prev_pos_d_clicks = 0
prev_submit_clicks = 0
prev_slide_ranger_clicks = 0
prev_button_flag = 0

slide_ranger_toggle = True

bin_data_length = 616
bin_data_type = 'd6B12fB4B3d4f4fHBHB12d8f3f3B7f10f3f6f6fBfB2df2d3f12f6f3f9f2B3f4B'
csv_header_list = ['rosTime', 'flightMode', 'ctrlDeviceStatus',
                   'fcMcMode', 'nSat', 'gpsFix', 'jobSeq',
                   'velNEDGps_mps_0', 'velNEDGps_mps_1', 'velNEDGps_mps_2',
                   'posNED_m_0', 'posNED_m_1', 'posNED_m_2',
                   'velNED_mps_0', 'velNED_mps_1', 'velNED_mps_2',
                   'rpy_deg_0', 'rpy_deg_1', 'rpy_deg_2',
                   'yawSpType',
                   'ctrlUser', 'ctrlStruct', 'ctrlSpType', 'ctrlOpType',
                   'ctrlSp_0', 'ctrlSp_1', 'ctrlSp_2',
                   'ctrlOp_0', 'ctrlOp_1', 'ctrlOp_2', 'yawSp_deg',
                   'rcRPYT_0', 'rcRPYT_1', 'rcRPYT_2', 'rcRPYT_3',
                   'gpsNSV', 'rtkHealthFlag', 'gpsFusedNSV', 'gpHealthStrength',
                   'posGPS_degE7_degE7_mm_0', 'posGPS_degE7_degE7_mm_1', 'posGPS_degE7_degE7_mm_2',
                   'posRTK_deg_deg_m_0', 'posRTK_deg_deg_m_1', 'posRTK_deg_deg_m_2',
                   'posGpsFused_rad_rad_m_0', 'posGpsFused_rad_rad_m_1', 'posGpsFused_rad_rad_m_2',
                   'posGP_deg_deg_m_0', 'posGP_deg_deg_m_1', 'posGP_deg_deg_m_2',
                   'StdJobLatCtrlPIDErrLatMix', 'StdJobLatCtrlPIDErrLatVis',
                   'StdJobLatCtrlPIDErrLatLidNorm', 'StdJobLatCtrlCmdVelLatIgain',
                   'StdJobLatCtrlCmdVelLatMix', 'StdJobLatCtrlPIDErrLatMixRate',
                   'StdJobLatCtrlPIDErrLatMixCov00', 'StdJobLatCtrlPIDErrLatMixCov11',
                   'velUVW_mps_0', 'velUVW_mps_1', 'velUVW_mps_2',
                   'AcWarnStat', 'AcHorWarnAC', 'AcVerWarnAC',
                   'AcXYZRel_m_0', 'AcXYZRel_m_1', 'AcXYZRel_m_2',
                   'AcHorWarnRange_m', 'AcHorWarnAngle_deg', 'AcVerWarnRange_m', 'AcVerWarnAngle_deg',
                   'LidarDist_m', 'LidarAngle_deg',
                   'LidarRaw_m_0', 'LidarRaw_m_1', 'LidarRaw_m_2', 'LidarRaw_m_3',
                   'LidarRaw_m_4', 'LidarRaw_m_5', 'LidarRaw_m_6', 'LidarRaw_m_7',
                   'llhVelCmd_1', 'llhVelCmd_0', 'llhVelCmd_2',
                   'velCtrlHdgI_0', 'velCtrlHdgI_1', 'velCtrlHdgI_2',
                   'posCtrlNEDI_0', 'posCtrlNEDI_1', 'posCtrlNEDI_2',
                   'gimbalRpyCmd_deg_0', 'gimbalRpyCmd_deg_1', 'gimbalRpyCmd_deg_2',
                   'gimbalRpy_deg_0', 'gimbalRpy_deg_1', 'gimbalRpy_deg_2',
                   'windStatus', 'windSpeed', 'windAngle', 'windQueryTime', 'windResponseTime',
                   'acousticTemp', 'tempQueryTime', 'tempResponseTime',
                   'accBody_mpss_0', 'accBody_mpss_1', 'accBody_mpss_2',
                   'trajUnitVectorT_0', 'trajUnitVectorT_1', 'trajUnitVectorT_2',
                   'trajUnitVectorN_0', 'trajUnitVectorN_1', 'trajUnitVectorN_2',
                   'trajUnitVectorB_0', 'trajUnitVectorB_1', 'trajUnitVectorB_2',
                   'trajVelCmdTNB_mps_0', 'trajVelCmdTNB_mps_1', 'trajVelCmdTNB_mps_2',
                   'StdJobLongPIDErr', 'StdJobLongPIDRate', 'StdJobLongPIDIgain',
                   'GuideModeLongPIDErr', 'GuideModeLongPIDRate', 'GuideModeLongPIDIgain',
                   'pqr_dps_0', 'pqr_dps_1', 'pqr_dps_2',
                   'rpdCmd_deg_deg_mps_0', 'rpdCmd_deg_deg_mps_1', 'rpdCmd_deg_deg_mps_2',
                   'velCmdHdg_mps_0', 'velCmdHdg_mps_1', 'velCmdHdg_mps_2',
                   'posCmdNED_m_0', 'posCmdNED_m_1', 'posCmdNED_m_2',
                   'missionType', 'jobType',
                   'bladeTravelDistance',
                   'trajTimeCur', 'trajTimeMax', 'pad_1', 'pad_2', 'pad_3', 'pad_4']


app.layout = html.Div([
    html.Div([
        dcc.ConfirmDialog(
            id='confirm_parsing_data',
        ),
        dcc.Upload(
            id='input_upload_data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')]),
            style={
                'width': '60%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'float': 'left'},
            multiple=True
        ),
        html.Div([
            html.A(html.Img(src=app.get_asset_url('nearthlab-logo-black-large.png'),
                            style={'height': '100%'}),
                   href='https://www.nearthlab.com/')
        ],
            style={
                'height': '40px',
                'textAlign': 'center',
                'padding-top': '20px'
        }),
    ],
        style={
            'display': 'inline'
    }
    ),
    html.Hr(style={
            'margin-bottom': '1.5rem'
    }),
    html.Div(
        id='output_log_status',
        children=''
    ),
    dcc.Tabs([
        dcc.Tab(label='2D Data Plot', children=[
            html.Div([
                html.Label([
                    dcc.Dropdown(
                        id='io_data_dropdown',
                        multi=True,
                        placeholder="Select Data"
                    ),
                ]),
                html.Label([
                    dcc.Dropdown(
                        id='io_data_dropdown_2',
                        multi=True,
                        placeholder="Select Data"
                    )
                ])]
            ),
            html.Div([
                html.Button('mission',
                            id='input_mission_button',
                            n_clicks=0,
                            style={'background-color': 'mistyrose'}),
                html.Button('gps',
                            id='input_gps_button',
                            n_clicks=0,
                            style={'background-color': 'mistyrose'}),
                html.Button('rpd_roll',
                            id='input_rpd_roll_button',
                            n_clicks=0,
                            style={'background-color': 'honeydew'}),
                html.Button('rpd_pitch',
                            id='input_rpd_pitch_button',
                            n_clicks=0,
                            style={'background-color': 'honeydew'}),
                html.Button('rpd_down',
                            id='input_rpd_down_button',
                            n_clicks=0,
                            style={'background-color': 'honeydew'}),
                html.Button('yaw',
                            id='input_yaw_button',
                            n_clicks=0,
                            style={'background-color': 'lavenderblush'}),
                html.Button('vel_x',
                            id='input_vel_x_button',
                            n_clicks=0,
                            style={'background-color': 'cornsilk'}),
                html.Button('vel_y',
                            id='input_vel_y_button',
                            n_clicks=0,
                            style={'background-color': 'cornsilk'}),
                html.Button('vel_z',
                            id='input_vel_z_button',
                            n_clicks=0,
                            style={'background-color': 'cornsilk'}),
                html.Button('pos_n',
                            id='input_pos_n_button',
                            n_clicks=0,
                            style={'background-color': 'azure'}),
                html.Button('pos_e',
                            id='input_pos_e_button',
                            n_clicks=0,
                            style={'background-color': 'azure'}),
                html.Button('pos_d',
                            id='input_pos_d_button',
                            n_clicks=0,
                            style={'background-color': 'azure'}),
                html.Button('slide_ranger: true',
                            id='input_slide_ranger_button',
                            n_clicks=0,
                            style={'float': 'right',
                                   'background-color': 'turquoise'})
            ]),
            dcc.Graph(id='graph_go')
        ]),
        dcc.Tab(label='3D Data Plot', children=[
            html.Div([
                dcc.Checklist(
                    id='output_select_data_checklist',
                    options=[
                        {'label': 'Flight Path', 'value': 'Flight_Path'},
                        {'label': 'Lidar Point Cloud From Octomap', 'value': 'Lidar_PC'}],
                    labelStyle={'display': 'inline-block'}
                )]
            ),
            dcc.Graph(id='graph_go_3d_pos')
        ])
    ])
])


def parse_contents(list_of_contents, list_of_names, list_of_dates):
    global df, df_pc, fcMcMode_index, fcMcMode_value, fcMcMode_color, \
        bin_data_length, bin_data_type, csv_header_list
    parsing_log = ''
    strNames = ''
    strDates = ''
    strDecoded = ''
    strFcLogVersion = ''
    strFcType = ''
    strUAVModel = ''
    strUserEnv = ''
    strMissionType = ''
    for contents, filename, date in zip(list_of_contents, list_of_names, list_of_dates):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_header_list_sorted = []
        if 'gdLog' in filename or 'aSensorLog' in filename:
            if 'csv' in filename:
                try:
                    tp = pd.read_csv(io.StringIO(decoded.decode('utf-8')), 
                                     iterator=True, chunksize=1000, low_memory=False)
                    df = pd.concat(tp, ignore_index=True)
                    parsing_log = parsing_log + 'gdLog csv file!\n'
                except Exception as e:
                    print('[parse_contents::read_gdlog_csv] ' + str(e))
            elif 'bin' in filename:
                try:
                    logging_rate = 50
                    if 'aSensorLog' in filename:
                        logging_rate = int(filename.split('.')[0].split('_')[-1])
                    with open(filename.split('.')[0] + '.csv', 'w', encoding='utf-8') as f_csv:
                        if chr(decoded[0]) == 'n':
                            strFcLogVersion = str(decoded[1])
                            print("New_Format v" + strFcLogVersion)
                            fcLogHeaderSize = decoded[3] << 8 | decoded[2]
                            fcLogTypeListSize = decoded[5] << 8 | decoded[4]
                            fcLogDataSize = decoded[7] << 8 | decoded[6]
                            fcLogHeader = decoded[8:8+fcLogHeaderSize].decode('ascii')
                            fcLogTypeList = decoded[8+fcLogHeaderSize:8+fcLogHeaderSize+fcLogTypeListSize].decode('ascii')

                            csv_header_list = fcLogHeader.split(",")
                            bin_data_type = '='+fcLogTypeList # Byte order: native, Size: standard
                            bin_data_length = fcLogDataSize

                            decoded = decoded[8+fcLogHeaderSize+fcLogTypeListSize:]
                        chunk = decoded[0:len(decoded)//bin_data_length*bin_data_length]
                        data_count = 0
                        wr = csv.writer(f_csv)
                        wr.writerow(csv_header_list)
                        for unpacked_chunk in struct.iter_unpack(bin_data_type, chunk):
                            wr.writerow(list(unpacked_chunk))
                            if data_count % 3000 == 0:
                                print("data_count: " + str(data_count))
                            data_count += 1
                        print('data_count: ' + str(data_count) +
                              ', total_time: ' + str(data_count/logging_rate) + ' s (' + str(logging_rate) + 'Hz)')
                        print('Saved: ' + f_csv.name)
                        parsing_log = parsing_log + 'data_count : ' + str(data_count) + \
                            '\ntotal_time : ' + str(data_count/logging_rate) + ' s (' + str(logging_rate) + 'Hz)\n'
                except Exception as e:
                    print('[parse_contents::binary_parser] ' + str(e))
                try:
                    df = pd.read_csv(filename.split('.')[0] + '.csv')
                    parsing_log = parsing_log + 'gdLog bin file!\n'
                except Exception as e:
                    print('[parse_contents::read_gdlog_bin] ' + str(e))
            elif 'xls' in filename:
                try:
                    df = pd.read_excel(io.BytesIO(decoded))
                    parsing_log = parsing_log + 'gdLog xls file!\n'
                except Exception as e:
                    print('[parse_contents::read_gdlog_xls] ' + str(e))

                # dataFrame Post-Processing
            try:
                if len(df) > 0:
                    df = df.drop([0])  # delete data with initial value
                    df = df.dropna(axis=0)  # delete data with NaN
                    df = df[df['rosTime'] > 1577840400] # Ignore data before 2020 January 1st Wednesday AM 1:00:00
                    df = df.reset_index(drop=True)
                    df.columns = df.columns.str.strip()

                if 'rosTime' in df.columns:
                    df['dateTime'] = pd.to_datetime(df['rosTime'], unit='s') + \
                        pd.DateOffset(hours=9)
                    df['diffTime'] = df['rosTime'].diff()
                    df['diffTimeHist'] = df['diffTime']

                if 'posNed_0' in df.columns:
                    print('old_format_csv')
                    df.rename(columns={
                        'velNedGps_0': 'velNEDGps_mps_0', 'velNedGps_1': 'velNEDGps_mps_1', 'velNedGps_2': 'velNEDGps_mps_2',
                        'posNed_0': 'posNED_m_0', 'posNed_1': 'posNED_m_1', 'posNed_2': 'posNED_m_2',
                        'velNed_0': 'velNED_mps_0', 'velNed_1': 'velNED_mps_1', 'velNed_2': 'velNED_mps_2',
                        'rpy_0': 'rpy_deg_0', 'rpy_1': 'rpy_deg_1', 'rpy_2': 'rpy_deg_2',
                        'yawSp': 'yawSp_deg',
                        'rcRoll': 'rcRPYT_0', 'rcPitch': 'rcRPYT_1', 'rcYaw': 'rcRPYT_2', 'rcThrottle': 'rcRPYT_3',
                        'GpHealth': 'GpHealthStrength',
                        'posGPS_0': 'posGPS_degE7_degE7_mm_0', 'posGPS_1': 'posGPS_degE7_degE7_mm_1', 'posGPS_2': 'posGPS_degE7_degE7_mm_2',
                        'posRTK_0': 'posRTK_deg_deg_m_0', 'posRTK_1': 'posRTK_deg_deg_m_1', 'posRTK_2': 'posRTK_deg_deg_m_2',
                        'posGpsFused_0': 'posGpsFused_rad_rad_m_0', 'posGpsFused_1': 'posGpsFused_rad_rad_m_1', 'posGpsFused_2': 'posGpsFused_rad_rad_m_2',
                        'posGp_0': 'posGP_deg_deg_m_0', 'posGp_1': 'posGP_deg_deg_m_1', 'posGp_2': 'posGP_deg_deg_m_2',
                        'errLatMix': 'StdJobLatCtrlPIDErrLatMix', 'errLatVis': 'StdJobLatCtrlPIDErrLatVis',
                        'errLatLid': 'StdJobLatCtrlPIDErrLatLidNorm', 'cmdLatVelIgain': 'StdJobLatCtrlCmdVelLatIgain',
                        'cmdLatVelMix': 'StdJobLatCtrlCmdVelLatMix', 'errLatMixRate': 'StdJobLatCtrlPIDErrLatMixRate',
                        'errLatMixCov00': 'StdJobLatCtrlPIDErrLatMixCov00', 'errLatMixCov11': 'StdJobLatCtrlPIDErrLatMixCov11',
                        'vbx': 'velUVW_mps_0', 'vby': 'velUVW_mps_1', 'vbz': 'velUVW_mps_2',
                        'AcXRel': 'AcXYZRel_m_0', 'AcYRel': 'AcXYZRel_m_1', 'AcZRel': 'AcXYZRel_m_2',
                        'AcHorWarnRange': 'AcHorWarnRange_m', 'AcHorWarnAngle': 'AcHorWarnAngle_deg',
                        'AcVerWarnRange': 'AcVerWarnRange_m', 'AcVerWarnAngle': 'AcVerWarnAngle_deg',
                        'LidarDist': 'LidarDist_m', 'LidarAngle': 'LidarAngle_deg',
                        'LidarRaw_0': 'LidarRaw_m_0', 'LidarRaw_1': 'LidarRaw_m_1', 'LidarRaw_2': 'LidarRaw_m_2', 'LidarRaw_3': 'LidarRaw_m_3',
                        'LidarRaw_4': 'LidarRaw_m_4', 'LidarRaw_5': 'LidarRaw_m_5', 'LidarRaw_6': 'LidarRaw_m_6', 'LidarRaw_7': 'LidarRaw_m_7',
                        'LongVelCmd': 'llhVelCmd_1', 'LatVelCmd': 'llhVelCmd_0', 'HeaveVelCmd': 'llhVelCmd_2',
                        'velCtrlI_u': 'velCtrlHdgI_0', 'velCtrlI_v': 'velCtrlHdgI_1', 'velCtrlI_d': 'velCtrlHdgI_2',
                        'posCtrlI_N': 'posCtrlNEDI_0', 'posCtrlI_E': 'posCtrlNEDI_1', 'posCtrlI_D': 'posCtrlNEDI_2',
                        'gimbalRollCmd': 'gimbalRpyCmd_deg_0', 'gimbalPitchCmd': 'gimbalRpyCmd_deg_1', 'gimbalYawCmd': 'gimbalRpyCmd_deg_2',
                        'gimbalRoll': 'gimbalRpy_deg_0', 'gimbalPitch': 'gimbalRpy_deg_1', 'gimbalYaw': 'gimbalRpy_deg_2',
                        'accBody_0': 'accBody_mpss_0', 'accBody_1': 'accBody_mpss_1', 'accBody_2': 'accBody_mpss_2',
                        'trajCmd_T': 'trajVelCmdTNB_mps_0', 'trajCmd_N': 'trajVelCmdTNB_mps_1', 'trajCmd_B': 'trajVelCmdTNB_mps_2',
                        'pqr_0': 'pqr_dps_0', 'pqr_1': 'pqr_dps_1', 'pqr_2': 'pqr_dps_2',
                        'rpdCmd_0': 'rpdCmd_deg_deg_mps_0', 'rpdCmd_1': 'rpdCmd_deg_deg_mps_1', 'rpdCmd_2': 'rpdCmd_deg_deg_mps_2',
                        'velCmdNav_0': 'velCmdHdg_mps_0', 'velCmdNav_1': 'velCmdHdg_mps_1', 'velCmdNav_2': 'velCmdHdg_mps_2',
                        'posCmdNed_0': 'posCmdNED_m_0', 'posCmdNed_1': 'posCmdNED_m_1', 'posCmdNed_2': 'posCmdNED_m_2'
                    },
                        inplace=True)

                if 'fcMcMode' in df.columns:
                    df.loc[df.fcMcMode == 0, 'strFcMcMode'] = 'RC'
                    df.loc[df.fcMcMode == 1, 'strFcMcMode'] = 'Guide'
                    df.loc[df.fcMcMode == 2, 'strFcMcMode'] = 'Auto'
                    df.loc[df.fcMcMode == 3, 'strFcMcMode'] = 'Boot'
                    df.loc[df.fcMcMode == 4, 'strFcMcMode'] = 'Standby'
                    df.loc[df.fcMcMode == 255, 'strFcMcMode'] = 'SafeHold'
                    df.loc[df.fcMcMode == 0, 'colorFcMcMode'] = 'yellow'
                    df.loc[df.fcMcMode == 1, 'colorFcMcMode'] = 'lightCoral'
                    df.loc[df.fcMcMode == 2, 'colorFcMcMode'] = 'turquoise'
                    df.loc[df.fcMcMode == 3, 'colorFcMcMode'] = 'lightPink'
                    df.loc[df.fcMcMode == 4, 'colorFcMcMode'] = 'blue'
                    df.loc[df.fcMcMode == 255, 'colorFcMcMode'] = 'red'
                    df['diffFcMcMode'] = df['fcMcMode'].diff()

                    fcMcMode_index = df.index[df['diffFcMcMode'] != 0].tolist()
                    fcMcMode_index = [fcMcMode_index[i] - fcMcMode_index[0] for i in range(len(fcMcMode_index))]
                    fcMcMode_index = np.append(fcMcMode_index, len(df)-1)
                    fcMcMode_value = df.iloc[fcMcMode_index].strFcMcMode.tolist()
                    fcMcMode_color = df.iloc[fcMcMode_index].colorFcMcMode.tolist()

                if 'fcType' in df.columns:
                    listFcType = ['Mockup', 'DJI', 'PX4']
                    strFcType = listFcType[df.fcType[0]]

                if 'UAVModel' in df.columns:
                    listUAVModel = ['M600', 'M210', 'M300', 'PX4']
                    strUAVModel = listUAVModel[df.UAVModel[0]]

                if 'IsSim' in df.columns:
                    listIsSim = ['Real', 'Sim']
                    strUserEnv = listIsSim[df.IsSim[0]]

                if 'userEnv' in df.columns:
                    listUserEnv = ['Real', 'SILS', 'HILS']
                    strUserEnv = listUserEnv[df.userEnv[0]]

                if 'missionType' in df.columns:
                    listMissionType = ['MISSION_TYPE_5_1', 'MISSION_TYPE_5_2', 'MISSION_TYPE_4_1',
                                       'MISSION_TYPE_4_2', 'MISSION_TYPE_4_3', 'MISSION_TYPE_WP',
                                       'MISSION_TYPE_3_1', 'MISSION_TYPE_3_2', 'MISSION_TYPE_6_1',
                                       'MISSION_TYPE_6_2', 'MISSION_TYPE_6_1_SP', 'MISSION_TYPE_6_2_SP',
                                       'MISSION_TYPE_HI']
                    if 'Standby' in fcMcMode_value:
                        strMissionType = listMissionType[df.missionType[fcMcMode_index[fcMcMode_value.index('Standby')]]]
                    elif 'Guide' in fcMcMode_value:
                        strMissionType = listMissionType[df.missionType[fcMcMode_index[fcMcMode_value.index('Guide')]]]

                if 'jobType' in df.columns:
                    df.loc[df.jobType == 0, 'strJobType'] = 'INIT'
                    df.loc[df.jobType == 1, 'strJobType'] = 'STANDARD'
                    df.loc[df.jobType == 2, 'strJobType'] = 'TURN'
                    df.loc[df.jobType == 3, 'strJobType'] = 'CAM'
                    df.loc[df.jobType == 4, 'strJobType'] = 'WAYPOINT'
                    df.loc[df.jobType == 5, 'strJobType'] = 'TRAJECTORY'
                    df.loc[df.jobType == 6, 'strJobType'] = 'REFSIGTEST'
                    df.loc[df.jobType == 7, 'strJobType'] = 'TAKEOFF'
                    df.loc[df.jobType == 8, 'strJobType'] = 'LAND'
                    df.loc[df.jobType == 9, 'strJobType'] = 'RTB'
                    df.loc[df.jobType == 10, 'strJobType'] = 'PATHPLANNING'
                    df.loc[df.jobType == 11, 'strJobType'] = 'SKELETONMODELESTIMATION'
                    df.loc[df.jobType == 12, 'strJobType'] = 'HOPPINGBLADESURFACE'
                    df.loc[df.jobType == 13, 'strJobType'] = 'BLADEFOLLOWING'
                    df.loc[df.jobType == 255, 'strJobType'] = 'NONE'

                if 'gpsFix' in df.columns:
                    df.loc[df.gpsFix == 0, 'strGpsFix'] = 'No_GPS'
                    df.loc[df.gpsFix == 1, 'strGpsFix'] = 'NO_FIX'
                    df.loc[df.gpsFix == 2, 'strGpsFix'] = '2D_FIX'
                    df.loc[df.gpsFix == 3, 'strGpsFix'] = '3D_FIX'
                    df.loc[df.gpsFix == 4, 'strGpsFix'] = '3D_DGPS/SBAS_AIDED'
                    df.loc[df.gpsFix == 5, 'strGpsFix'] = '3D_RTK_FLOAT'
                    df.loc[df.gpsFix == 6, 'strGpsFix'] = '3D_RTK_FIXED'
                    df.loc[df.gpsFix == 7, 'strGpsFix'] = '3D_STATIC'
                    df.loc[df.gpsFix == 8, 'strGpsFix'] = '3D_PPP'

                if 'ctrlStruct' in df.columns:
                    df.loc[df.ctrlStruct == 0, 'strCtrlStruct'] = 'CTRL_STRUCT_NONE'
                    df.loc[df.ctrlStruct == 1, 'strCtrlStruct'] = 'VELI_PID0_ATTI_RPD'
                    df.loc[df.ctrlStruct == 2, 'strCtrlStruct'] = 'VELI_PID0CA_ATTI_RPD'
                    df.loc[df.ctrlStruct == 3, 'strCtrlStruct'] = 'VELB_PID0_ATTI_RPD'
                    df.loc[df.ctrlStruct == 4, 'strCtrlStruct'] = 'VELB_PID0CA_ATTI_RPD'
                    df.loc[df.ctrlStruct == 5, 'strCtrlStruct'] = 'VELB_0_VELB'
                    df.loc[df.ctrlStruct == 6, 'strCtrlStruct'] = 'VELB_0CA_VELB'
                    df.loc[df.ctrlStruct == 7, 'strCtrlStruct'] = 'VELI_0CA_VELI'
                    df.loc[df.ctrlStruct == 8, 'strCtrlStruct'] = 'POSI_PID0CA_VELI'
                    df.loc[df.ctrlStruct == 9, 'strCtrlStruct'] = 'POSI_PID0CA_ATTI'
                    df.loc[df.ctrlStruct == 10, 'strCtrlStruct'] = 'POSBVELB_PID0CA_ATTI'
                    df.loc[df.ctrlStruct == 11, 'strCtrlStruct'] = 'POSI_0_POSI'
                    df.loc[df.ctrlStruct == 12, 'strCtrlStruct'] = 'POSI_CA_POSI'
                    df.loc[df.ctrlStruct == 13, 'strCtrlStruct'] = 'TRAJ_PID0AC_ATTI_RPD'
                    df.loc[df.ctrlStruct == 14, 'strCtrlStruct'] = 'ATTI_0_ATTI'

                if 'ctrlSpType' in df.columns:
                    df.loc[df.ctrlSpType == 0, 'strCtrlSpType'] = 'CTRL_VECTYPE_NONE'
                    df.loc[df.ctrlSpType == 1, 'strCtrlSpType'] = 'LLH_POS'
                    df.loc[df.ctrlSpType == 2, 'strCtrlSpType'] = 'NEDABS_POS'
                    df.loc[df.ctrlSpType == 3, 'strCtrlSpType'] = 'NEALTABS_POS'
                    df.loc[df.ctrlSpType == 4, 'strCtrlSpType'] = 'NEALTREL_POS'
                    df.loc[df.ctrlSpType == 5, 'strCtrlSpType'] = 'XYALT_POS'
                    df.loc[df.ctrlSpType == 6, 'strCtrlSpType'] = 'XYD_POS'
                    df.loc[df.ctrlSpType == 7, 'strCtrlSpType'] = 'NED_VEL'
                    df.loc[df.ctrlSpType == 8, 'strCtrlSpType'] = 'UVW_VEL'
                    df.loc[df.ctrlSpType == 9, 'strCtrlSpType'] = 'EULER_ATT'
                    df.loc[df.ctrlSpType == 10, 'strCtrlSpType'] = 'TRAJ_VEC'
                    df.loc[df.ctrlSpType == 11, 'strCtrlSpType'] = 'XYZ_VEL'

                if 'ctrlOpType' in df.columns:
                    df.loc[df.ctrlOpType == 0, 'strCtrlOpType'] = 'CTRL_VECTYPE_NONE'
                    df.loc[df.ctrlOpType == 1, 'strCtrlOpType'] = 'LLH_POS'
                    df.loc[df.ctrlOpType == 2, 'strCtrlOpType'] = 'NEDABS_POS'
                    df.loc[df.ctrlOpType == 3, 'strCtrlOpType'] = 'NEALTABS_POS'
                    df.loc[df.ctrlOpType == 4, 'strCtrlOpType'] = 'NEALTREL_POS'
                    df.loc[df.ctrlOpType == 5, 'strCtrlOpType'] = 'XYALT_POS'
                    df.loc[df.ctrlOpType == 6, 'strCtrlOpType'] = 'XYD_POS'
                    df.loc[df.ctrlOpType == 7, 'strCtrlOpType'] = 'NED_VEL'
                    df.loc[df.ctrlOpType == 8, 'strCtrlOpType'] = 'UVW_VEL'
                    df.loc[df.ctrlOpType == 9, 'strCtrlOpType'] = 'EULER_ATT'
                    df.loc[df.ctrlOpType == 10, 'strCtrlOpType'] = 'TRAJ_VEC'
                    df.loc[df.ctrlOpType == 11, 'strCtrlOpType'] = 'XYZ_VEL'

                if 'yawOpType' in df.columns:
                    df.loc[df.yawOpType == 0, 'strYawOpType'] = 'ANGLE_REL'
                    df.loc[df.yawOpType == 1, 'strYawOpType'] = 'ANGLE_ABS'
                    df.loc[df.yawOpType == 2, 'strYawOpType'] = 'RATE'
                    df.loc[df.yawOpType == 3, 'strYawOpType'] = 'FOWARD'

                df_header_list_sorted = sorted(df.columns.tolist())
            except Exception as e:
                print('[parse_contents::data_post_processing] ' + str(e))
        elif True: #'pointCloud' in filename:
            if 'csv' in filename:
                try:
                    np_pc = np.loadtxt(io.StringIO(decoded.decode('utf-8')),
                                       delimiter=',')
                    np_pc = np_pc.astype(np.float)
                    np_pc = np_pc.reshape(-1, 3)
                    df_pc = pd.DataFrame(np_pc, columns=['x', 'y', 'z'])
                    parsing_log = parsing_log + 'pointCloud csv file!\n'
                except Exception as e:
                    print('[parse_contents::read_pointCloud_csv] ' + str(e))
        try:
            strNames = strNames + filename + '\n'
            strDates = strDates + \
                str(datetime.datetime.fromtimestamp(date)) + '\n'
            strDecoded = strDecoded + str(decoded[0:100]) + '...\n'
        except Exception as e:
            print('[parse_contents::make_string] ' + str(e))
        try:
            childrenLogStatus = html.P(
                children=['[fcLogVersion] ㅤ',
                          html.B(strFcLogVersion),
                          ' ㅤㅤㅤㅤ [fcType] ㅤ',
                          html.B(strFcType),
                          ' ㅤㅤㅤㅤ [UAVModel] ㅤ',
                          html.B(strUAVModel),
                          ' ㅤㅤㅤㅤ [userEnv] ㅤ',
                          html.B(strUserEnv),
                          ' ㅤㅤㅤㅤ [missionType] ㅤ',
                          html.B(strMissionType)]
            )
        except Exception as e:
            print('[parse_contents::make_log_status] ' + str(e))
        confirm_msg = '[Parsing Log]\n' + parsing_log + \
                      '\n[File Names]\n' + strNames + \
                      '\n[Raw Contents]\n' + strDecoded + \
                      '\n[Do you want to use only the Guide/Auto Data?]\n'
    return confirm_msg, df_header_list_sorted, childrenLogStatus


def reset_pre_button_clicks():
    global prev_mission_clicks, prev_gps_clicks
    global prev_rpd_roll_clicks, prev_rpd_pitch_clicks, prev_rpd_down_clicks
    global prev_yaw_clicks
    global prev_vel_x_clicks, prev_vel_y_clicks, prev_vel_z_clicks
    global prev_pos_n_clicks, prev_pos_e_clicks, prev_pos_d_clicks

    prev_mission_clicks = 0
    prev_gps_clicks = 0
    prev_rpd_roll_clicks = 0
    prev_rpd_pitch_clicks = 0
    prev_rpd_down_clicks = 0
    prev_yaw_clicks = 0
    prev_vel_x_clicks = 0
    prev_vel_y_clicks = 0
    prev_vel_z_clicks = 0
    prev_pos_n_clicks = 0
    prev_pos_e_clicks = 0
    prev_pos_d_clicks = 0


@app.callback(
    Output('io_data_dropdown', 'options'),
    Output('io_data_dropdown_2', 'options'),
    Output('confirm_parsing_data', 'displayed'),
    Output('confirm_parsing_data', 'message'),
    Output('output_log_status', 'children'),
    Output('input_mission_button', 'n_clicks'),
    Output('input_gps_button', 'n_clicks'),
    Output('input_rpd_roll_button', 'n_clicks'),
    Output('input_rpd_pitch_button', 'n_clicks'),
    Output('input_rpd_down_button', 'n_clicks'),
    Output('input_yaw_button', 'n_clicks'),
    Output('input_vel_x_button', 'n_clicks'),
    Output('input_vel_y_button', 'n_clicks'),
    Output('input_vel_z_button', 'n_clicks'),
    Output('input_pos_n_button', 'n_clicks'),
    Output('input_pos_e_button', 'n_clicks'),
    Output('input_pos_d_button', 'n_clicks'),
    Input('input_upload_data', 'contents'),
    State('input_upload_data', 'filename'),
    State('input_upload_data', 'last_modified')
)
def update_data_upload(list_of_contents, list_of_names, list_of_dates):
    global prev_button_flag
    if list_of_contents is not None:
        confirm_msg, df_header_list_sorted, children_LogStatus = \
            parse_contents(list_of_contents, list_of_names, list_of_dates)
        options = [{'label': df_header, 'value': df_header}
                   for df_header in df_header_list_sorted]
        reset_pre_button_clicks()
        button_reset_list = [0,0,0,0,0,0,0,0,0,0,0,0]
        button_reset_list[prev_button_flag] = 1
        return options, options, True, confirm_msg, children_LogStatus,\
            button_reset_list[0], button_reset_list[1], button_reset_list[2],\
            button_reset_list[3], button_reset_list[4], button_reset_list[5],\
            button_reset_list[6], button_reset_list[7], button_reset_list[8],\
            button_reset_list[9], button_reset_list[10], button_reset_list[11]


@app.callback(
    Output('confirm_parsing_data', 'cancel_n_clicks'),  # dummy output
    Input('confirm_parsing_data', 'submit_n_clicks')
)
def update_df_data(submit_clicks):
    global df, fcMcMode_index, fcMcMode_value, fcMcMode_color
    global prev_submit_clicks

    try:
        if submit_clicks != prev_submit_clicks:
            for idx in range(len(fcMcMode_index)-1):
                if fcMcMode_value[idx] == 'Guide':
                    cut_begin_idx = idx
                    cut_begin = fcMcMode_index[idx]
                    break
            for idx in reversed(range(len(fcMcMode_index)-1)):
                if fcMcMode_value[idx] != 'RC':
                    cut_end_idx = idx+1
                    cut_end = fcMcMode_index[idx+1]
                    break
            df = df[cut_begin:cut_end]
            
            if ('posNED_m_0' in df.columns):
                    # Ignore position data under -1,000m
                df = df[df['posNED_m_0'] > -1000]
                df = df[df['posNED_m_1'] > -1000]
                df = df[df['posNED_m_2'] > -1000]
                # Ignore position data over +1,000m
                df = df[df['posNED_m_0'] < 1000]
                df = df[df['posNED_m_1'] < 1000]
                df = df[df['posNED_m_2'] < 1000]

            df = df.reset_index(drop=True)

            fcMcMode_index = fcMcMode_index[cut_begin_idx:cut_end_idx] - fcMcMode_index[cut_begin_idx]
            fcMcMode_index = np.append(fcMcMode_index, len(df)-1)
            fcMcMode_value = fcMcMode_value[cut_begin_idx:cut_end_idx]
            fcMcMode_color = fcMcMode_color[cut_begin_idx:cut_end_idx]

            prev_submit_clicks = submit_clicks
    except Exception as e:
        print('[update_df_data::cut_data] ' + str(e))
    return 0


@app.callback(
    Output('input_slide_ranger_button', 'children'),
    Input('input_slide_ranger_button', 'children'),
    Input('input_slide_ranger_button', 'n_clicks')
)
def update_graph_data(prev_slide_ranger_children, slide_ranger_clicks):
    global prev_slide_ranger_clicks, slide_ranger_toggle
    strSlideRanger = prev_slide_ranger_children

    if prev_slide_ranger_clicks != slide_ranger_clicks:
        slide_ranger_toggle = not slide_ranger_toggle
        strSlideRanger = 'slide_ranger: ' + str(slide_ranger_toggle)
        prev_slide_ranger_clicks = slide_ranger_clicks
    return strSlideRanger


@app.callback(
    Output('graph_go', 'figure'),
    Output('graph_go', 'config'),
    Input('io_data_dropdown', 'value'),
    Input('io_data_dropdown_2', 'value'),
    Input('input_mission_button', 'n_clicks'),
    Input('input_gps_button', 'n_clicks'),
    Input('input_rpd_roll_button', 'n_clicks'),
    Input('input_rpd_pitch_button', 'n_clicks'),
    Input('input_rpd_down_button', 'n_clicks'),
    Input('input_yaw_button', 'n_clicks'),
    Input('input_vel_x_button', 'n_clicks'),
    Input('input_vel_y_button', 'n_clicks'),
    Input('input_vel_z_button', 'n_clicks'),
    Input('input_pos_n_button', 'n_clicks'),
    Input('input_pos_e_button', 'n_clicks'),
    Input('input_pos_d_button', 'n_clicks')
)
def update_graph_data(df_header, df_header_2,
                      mission_clicks, gps_clicks,
                      rpd_roll_clicks, rpd_pitch_clicks, rpd_down_clicks,
                      yaw_clicks,
                      vel_x_clicks, vel_y_clicks, vel_z_clicks,
                      pos_n_clicks, pos_e_clicks, pos_d_clicks):
    global df, fcMcMode_index, fcMcMode_value, fcMcMode_color
    global prev_mission_clicks, prev_gps_clicks
    global prev_rpd_roll_clicks, prev_rpd_pitch_clicks, prev_rpd_down_clicks
    global prev_yaw_clicks
    global prev_vel_x_clicks, prev_vel_y_clicks, prev_vel_z_clicks
    global prev_pos_n_clicks, prev_pos_e_clicks, prev_pos_d_clicks
    global prev_button_flag

    if prev_mission_clicks != mission_clicks:
        df_header = ['jobSeq']
        df_header_2 = ['strJobType']
        prev_mission_clicks = mission_clicks
        prev_button_flag = 0
    elif prev_gps_clicks != gps_clicks:
        df_header = ['nSat', 'gpsNSV']
        df_header_2 = ['strGpsFix']
        prev_gps_clicks = gps_clicks
        prev_button_flag = 1
    elif prev_rpd_roll_clicks != rpd_roll_clicks:
        df_header = ['rpy_deg_0', 'rpdCmd_deg_deg_mps_0']
        df_header_2 = ['strCtrlStruct']
        prev_rpd_roll_clicks = rpd_roll_clicks
        prev_button_flag = 2
    elif prev_rpd_pitch_clicks != rpd_pitch_clicks:
        df_header = ['rpy_deg_1', 'rpdCmd_deg_deg_mps_1']
        df_header_2 = ['strCtrlStruct']
        prev_rpd_pitch_clicks = rpd_pitch_clicks
        prev_button_flag = 3
    elif prev_rpd_down_clicks != rpd_down_clicks:
        df_header = ['velUVW_mps_2', 'velCmdUVW_mps_2']
        df_header_2 = ['strCtrlStruct']
        prev_rpd_down_clicks = rpd_down_clicks
        prev_button_flag = 4
    elif prev_yaw_clicks != yaw_clicks:
        df_header = ['rpy_deg_2', 'yawSp_deg']
        df_header_2 = ['strCtrlStruct']
        prev_yaw_clicks = yaw_clicks
        prev_button_flag = 5
    elif prev_vel_x_clicks != vel_x_clicks:
        df_header = ['velHdg_mps_0', 'velCmdHdg_mps_0']
        df_header_2 = ['strCtrlStruct']
        prev_vel_x_clicks = vel_x_clicks
        prev_button_flag = 6
    elif prev_vel_y_clicks != vel_y_clicks:
        df_header = ['velHdg_mps_1', 'velCmdHdg_mps_1']
        df_header_2 = ['strCtrlStruct']
        prev_vel_y_clicks = vel_y_clicks
        prev_button_flag = 7
    elif prev_vel_z_clicks != vel_z_clicks:
        df_header = ['velHdg_mps_2', 'velCmdHdg_mps_2']
        df_header_2 = ['strCtrlStruct']
        prev_vel_z_clicks = vel_z_clicks
        prev_button_flag = 8
    elif prev_pos_n_clicks != pos_n_clicks:
        df_header = ['posNED_m_0', 'posCmdNED_m_0']
        df_header_2 = ['strCtrlStruct']
        prev_pos_n_clicks = pos_n_clicks
        prev_button_flag = 9
    elif prev_pos_e_clicks != pos_e_clicks:
        df_header = ['posNED_m_1', 'posCmdNED_m_1']
        df_header_2 = ['strCtrlStruct']
        prev_pos_e_clicks = pos_e_clicks
        prev_button_flag = 10
    elif prev_pos_d_clicks != pos_d_clicks:
        df_header = ['posNED_m_2', 'posCmdNED_m_2']
        df_header_2 = ['strCtrlStruct']
        prev_pos_d_clicks = pos_d_clicks
        prev_button_flag = 11

    figure = make_subplots(
        specs=[[{"secondary_y": True}]],
        shared_xaxes=True
    )
    figure.update_layout(height=675,
                         margin=dict(r=20, b=10, l=10, t=10))
    x_title = 'dateTime'
    try:
        if 'diffTimeHist' in df_header:
            figure.add_trace(go.Histogram(x=df['diffTimeHist']))
            config = dict({'displaylogo': False})
            return figure, config
        else:
            for y_title in df_header:
                figure.add_trace(go.Scatter(
                    x=df[x_title], y=df[y_title], name=y_title,
                    mode='lines',
                    line=dict(width=3)),
                    secondary_y=False
                )
    except Exception as e:
        print('[update_graph_data::df_header_trace] ' + str(e))
    try:
        for y_title in df_header_2:
            figure.add_trace(go.Scatter(
                x=df[x_title], y=df[y_title], name=y_title,
                mode='lines',
                line=dict(width=3)),
                secondary_y=True
            )
    except Exception as e:
        print('[update_graph_data::df_header2_trace] ' + str(e))
    try:
        plot_secondary_y = False
        if (df_header is None) or (len(df_header) == 0):
            plot_secondary_y = True
        elif (df_header_2 is None) or (len(df_header_2) == 0):
            plot_secondary_y = False
        for idx in range(len(fcMcMode_index)-1):
            figure.add_vrect(
                x0=df.iloc[fcMcMode_index[idx]].dateTime,
                x1=df.iloc[fcMcMode_index[idx+1]].dateTime,
                line_width=0,
                annotation_text=fcMcMode_value[idx],
                annotation_position="bottom left" if fcMcMode_value[idx]=='Guide' else "top left",
                fillcolor=fcMcMode_color[idx],
                layer="below",
                opacity=0.2,
                secondary_y=plot_secondary_y
            )
    except Exception as e:
        print('[update_graph_data::fcMcMode_vrect] ' + str(e))
    figure.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=slide_ranger_toggle,
                thickness=0.1
            )
        )
    )
    config = dict({'displaylogo': False,
                   'scrollZoom': True
                   })
    return figure, config


@app.callback(
    Output("graph_go_3d_pos", "figure"),
    Output("graph_go_3d_pos", "config"),
    [Input("output_select_data_checklist", "value")]
)
def update_3d_graph_data(plot_data_value):
    global df
    figure_3d = go.Figure()
    figure_3d.update_layout(scene=dict(
        xaxis_title='y_East',
        yaxis_title='x_North',
        zaxis_title='-z_Up',
        aspectmode='data'),
        height=630,
        margin=dict(r=20, b=10, l=10, t=10))
    if 'Flight_Path' in plot_data_value:
        try:
            for job_idx in df['jobSeq'].unique():
                df_jobSeq = df[df['jobSeq'] == job_idx]
                figure_3d.add_trace(go.Scatter3d(
                    x=df_jobSeq['posNED_m_1'],
                    y=df_jobSeq['posNED_m_0'],
                    z=-df_jobSeq['posNED_m_2'],
                    name='Flight Path (jobSeq = ' + str(job_idx) + ')',
                    mode='lines',
                    line=dict(color=-df_jobSeq['rosTime'],
                              colorscale='Viridis', width=6),
                    text=df_jobSeq['strFcMcMode'],
                    customdata=df_jobSeq['dateTime'],
                    hovertemplate=
                        'Time: <b>%{customdata}</b><br>' +
                        'fcMcMode: <b>%{text}</b><br>' +
                        'X: <b>%{x}</b><br>' +
                        'Y: <b>%{y}</b><br>' +
                        'Z: <b>%{z}</b>'
                ))
        except Exception as e:
            print('[update_3d_graph_data::Flight_Path] ' + str(e))
    if 'Lidar_PC' in plot_data_value:
        try:
            figure_3d.add_trace(go.Scatter3d(
                # x=df_pc['y'], y=df_pc['x'], z=-df_pc['z'],
                x=df_pc['x'], y=df_pc['y'], z=df_pc['z'],
                name='Lidar Point Cloud',
                mode='markers',
                marker=dict(size=2)))
            figure_3d.update_layout(scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='data'), 
                height=1200)
        except Exception as e:
            print('[update_3d_graph_data::Lidar_PC] ' + str(e))
    config_3d = dict({'displaylogo': False})
    return figure_3d, config_3d


if __name__ == '__main__':
    host_address='127.0.0.1'
    if len(sys.argv) > 1:
        host_address=sys.argv[1]
    while(True):
        try:
            app.run_server(debug=True, host=host_address)
        except Exception as e:
            print('[__main__::run_server] ' + str(e))
