"""
File: tram_energy_savings_app.py
Author: Florian Schubert, Clara Tillous Oliva, Yash Dubey
Date: November 2023
Description: A Streamlit app to estimate energy savings in trams with different heating systems and set point temperatures.
"""

import streamlit as st 
import pandas as pd
import numpy as np
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt

##### CONSTANTS #####

LIST_OF_CITIES = ["Zurich", "Bern", "Basel", "Geneva", "Neuchatel"]
#LIST_OF_CITIES = ["Zurich"]

C_P_AIR = 1004 # mass-specific heat capacity of air (at T=20°C, p=1050hPa) [J/kgK]
RHO_AIR = 1.275 # density of air (at T=20°C, p=1050hPa) [kg/m³]
HEAT_PERSON = 116 # average heat generation by on person (under sitting conditions) [W]
MONTH_DAYS_BEGIN = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366] # day in year of monthly beginning
MONTH_DAYS_MID = [16, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350] # day in year of mid-month
TIME_SHIFT_TO_UTC = +1 # UTC to MST

#CSV_PATH_TEMPERATURE = 'TempZurichHourly.csv'
CSV_PATH_TEMPERATURE = 'TempSwitzerland.csv'
#CSV_PATH_TEMPERATURE = "/Users/yash/Documents/GitHub/temp_trim/TempSwitzerland.csv"
#CSV_PATH_SOLAR_IRRADIATION = 'SRZurichHourly.csv'
CSV_PATH_SOLAR_IRRADIATION = 'SRBSwitzerland.csv'
#CSV_PATH_SOLAR_IRRADIATION = "/Users/yash/Documents/GitHub/temp_trim/SRBSwitzerland.csv"



##### THERMODYNAMICS #####

##### THERMODYNAMICS #####

# function calc_area_convection
# parameters: tram dimensions [m]
# returns total tram effective convection area [m²]
def calc_area_convection(length, width, height):
    return 2*(length*height + length*width + width*height)


# function volume_to_mass_flow
# parameter: volume air flow [m³/s]
# returns mass air flow [kg/s]
def volume_to_mass_flow(volume_flow):
    return RHO_AIR*volume_flow


# function generate_heat_pump_hierarchy
# parameter: array of coefficient of performance for heat pumps (one entry per heat pump) [-]
# returns list with heat pump indices sorted by decreasing COP
def generate_heat_pump_hierarchy(cops):
    return (len(cops) - 1 - np.argsort(cops))


# function calculate_operation_days_per_month
# parameter: begin and end days with month and day (1=January, 12=December | 1=first day in month)
# returns list with operation days per months (1 to 12 for January to December with indices 0 to 11)
def calculate_operation_days_per_month(begin_month, begin_day, end_month, end_day):
    day_begin = MONTH_DAYS_BEGIN[begin_month-1] + begin_day - 1
    day_end = MONTH_DAYS_BEGIN[end_month-1] + end_day - 1

    operation_days = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for m in range(0, 12):
        if day_begin <= day_end:
            day_begin_m = max(day_begin - 1, MONTH_DAYS_BEGIN[m])
            day_end_m = min(day_end, MONTH_DAYS_BEGIN[m + 1] - 1)
            operation_days[m] = max(0, day_end_m - day_begin_m + 1)
        else:
            day_begin_m = max(day_end, MONTH_DAYS_BEGIN[m])
            day_end_m = min(day_begin - 1, MONTH_DAYS_BEGIN[m + 1] - 1)
            operation_days[m] = (MONTH_DAYS_BEGIN[m + 1] - MONTH_DAYS_BEGIN[m]) - max(0, day_end_m - day_begin_m + 1)
    return operation_days


# function load_temperature_data
# parameter: location name
# returns array with temperature data (2d: 12 months, 24 hours)
def load_temperature_data(location):
    data = pd.read_csv(CSV_PATH_TEMPERATURE)
    T = [0]*12
    for m in range(0, 12):
        T[m] = data[location + '_' + str(m+1)].values
    return T


# function load_solar_irradiation_data
# parameter: location name
# returns array with temperature data (2d: 12 months, 24 hours)
def load_solar_irradiation_data(location):
    data = pd.read_csv(CSV_PATH_SOLAR_IRRADIATION)
    I_sun = [0]*12
    for m in range(0, 12):
        I_sun[m] = data[location + '_' + str(m+1)].values
    return I_sun



# tram data class
@dataclass
class Tram:
    # class init function
    # parameters: see comments in function, except for
    #   - volume flows [m³/s] instead of mass flows
    def __init__(self, name, passenger_number, volume_flow_ventilation, volume_flow_doors, length, width, height,
                 k_chassis, heat_auxiliary, heat_resistive, heat_pumps, cop_pumps, tram_count, tram_fraction_operational):
        # single vehicle specification
        self.name = name # name
        self.passenger_number = passenger_number # total average passenger number per tram [-]
        self.mass_flow_ventilation = volume_to_mass_flow(volume_flow_ventilation) # total average mass flow by ventilation system [kg/s]
        self.mass_flow_doors = volume_to_mass_flow(volume_flow_doors) # total average mass flow by open doors [kg/s]
        self.area_convection = calc_area_convection(length, width, height) # effective area for convective heat losses [m²]
        self.length = length  # tram length [m]
        self.width = width  # tram width [m]
        self.height = height  # tram height [m]
        self.k_chassis = k_chassis # effective average heat transfer coefficient for convective losses [W/m²K]
        self.heat_auxiliary = heat_auxiliary # total average heat generation by auxiliary devices in tram [W]
        self.heat_resistive = heat_resistive # total maximum resistive heat generation [W]
        self.heat_pumps = heat_pumps # array of maximum heat pump heat generation (one entry per heat pump) [W]
        self.cop_pumps = cop_pumps # array of coefficient of performance for heat pumps (one entry per heat pump) [-]
        self.heat_pump_hierarchy = generate_heat_pump_hierarchy(cop_pumps) # list of heat pump indices sorted by decreasing COP
        # fleet specification for identical vehicle type
        self.tram_count = tram_count # number of identical trams of the above specified tram type [-]
        self.tram_fraction_operational = tram_fraction_operational # fraction of average operational trams of the above specified tram type [-]


# climate schedule data class
@dataclass
class ClimateSchedule:
    # class init function
    # parameters: see comments in function, except for
    #   - begin and day by months and days in month (1=January, 12=December | 1=first day in month)
    #   - daily average normal sun irradiative energy [Wh/m²] instead of average sun intensity
    def __init__(self, begin_month, begin_day, end_month, end_day, t_begin, t_end, latitude, T_environment, I_sun, f_sun,
                 cost_of_electricity):
        # list with operation days per months (1 to 12 for January to December with indices 0 to 11)
        self.operation_days = calculate_operation_days_per_month(begin_month, begin_day, end_month, end_day)

        self.t_begin = t_begin # daily average operation begin time [h]
        if t_end < t_begin:
            self.t_end = t_end + 24
        else:
            self.t_end = t_end
        # daily average operation end time [h]

        self.latitude = latitude # latitude or location [°]
        self.T_environment = T_environment # two-dimensional array of (12) monthly and hourly (24) average environment temperatures [°C]
        self.I_sun = I_sun # two-dimensional array of (12) monthly and hourly (24) average normal solar irradiation [W/m²]

        self.f_sun = f_sun # yearly average un-shaded sun fraction [-]

        self.cost_of_electricity = cost_of_electricity # cost of electricity [CHF/Wh]


# consumption data class
@dataclass
class Consumption:
    # class init function
    # parameters: see comments in function
    def __init__(self, heat_total, electricity_total, electricity_cost_total, savings_total, T_setpoint_temperatures,
                 months, hours, tram_names, heat_instantaneous, electricity_instantaneous):
        self.heat_total = heat_total # array of total heat energy by heaters of all trams during whole interval per T_setpoint [Wh]
        self.electricity_total = electricity_total # array of total electricity consumption by heaters of all trams during whole interval per T_setpoint [Wh]
        self.electricity_cost_total = electricity_cost_total # array of total cost of electricity consumption by heaters of all trams during whole interval per T_setpoint [CHF]
        self.savings_total = savings_total # total savings as text (relative to highest T_setpoint)
        self.T_setpoint_temperatures = T_setpoint_temperatures  # array with set-point temperatures used for calculations

        self.months = months # array of analyzed months
        self.hours = hours  # array of analyzed hours
        self.tram_names = tram_names  # array of tram names

        self.heat_instantaneous = heat_instantaneous # five dimensional array with instantaneous heat values
        self.electricity_instantaneous = electricity_instantaneous  # five dimensional array with instantaneous electricity consumption values
        # dimension 0: set-point temperature (according to array T_setpoint_temperatures)
        # dimension 1: month (according to array months)
        # dimension 2: hour (according to array hours)
        # dimension 3: tram (according to array tram_names)
        # dimension 4: - instantaneous heat value array (0: total heating demand, 1: sun radiative heating,
        #                                                   2: passenger heating, 3: auxiliary device heating,
        #                                                   4: convective losses, 5: ventilation losses, 6: open door losses)
        #              - instantaneous electricity array (first entry resistive heater power, electric heat pump powers following)


# heating power error class
class ExceededHeatingPowerError(Exception):
    def __init__(self, message):
        self.message = message



# function calculate_absorption_area
# todo commenting
def calculate_absorption_area(tram_length, tram_width, tram_height, latitude, month, hour):
    delta = 23.45/180*math.pi * math.sin(2*math.pi/365*(284+MONTH_DAYS_MID[month]))
    omega = 15/180*math.pi * (hour-12)
    latitude_rad = math.pi/180 * latitude
    phi_zenith = math.acos(math.cos(latitude_rad)*math.cos(delta)*math.cos(omega) + math.sin(latitude_rad)*math.sin(delta))
    area_normal = math.sin(phi_zenith)/math.pi * tram_height*(tram_width+tram_length)
    return area_normal


# function calculate_heating_instantaneous:
#   thermodynamical model of instantaneous heat generation in one tram
# parameters:
#   0, tram:            Class tram object
#   1, T_tram:          Set-point temperature for tram cabin [°C]
#   2, T_environment:   Temperature or environment [°C]
#   3, I_sun:           Average (normal) sun intensity [W/m²]
#   4, f_sun:           Sun fraction [-]
#   5, A_abs:           (Normal) absorption area for sun [m²]
# returns: Instantaneous heat array [W] (0: total heating demand, 1: sun radiative heating, 2: passenger heating,
# 3: auxiliary device heating, 4: convective losses, 5: ventilation losses, 6: open door losses)
def calculate_heating_instantaneous(tram, T_tram, T_environment, I_sun, f_sun, A_abs):
    heat_sun = A_abs * I_sun * f_sun
    heat_passenger = tram.passenger_number * HEAT_PERSON
    heat_auxiliary_devices = tram.heat_auxiliary
    heat_loss_convection = tram.area_convection * tram.k_chassis * (T_tram - T_environment)
    heat_loss_ventilation =  tram.mass_flow_ventilation * C_P_AIR * (T_tram - T_environment)
    heat_loss_doors =  tram.mass_flow_doors * C_P_AIR * (T_tram - T_environment)
    heating_demand = max(0, ((heat_loss_convection + heat_loss_ventilation + heat_loss_doors)
                      - (heat_sun + heat_passenger + heat_auxiliary_devices)))
    return [heating_demand, heat_sun, heat_passenger, heat_auxiliary_devices, heat_loss_convection,
            heat_loss_ventilation, heat_loss_doors]
    # todo references to equation numbers in thermodynamics paper


# function calculate_total_consumption_and_savings:
#   consumption and savings total (all trams in all fleets during total interval)
# parameters:
#   0, tram:            Class tram objects
#   1, heat_demand:     Heat demand [W]
# returns: Instantaneous electricity demand per device [W] (first entry resistive heater power,
#           electric heat pump powers following)
def model_electricity_consumption(tram, heat_demand):
    electricity_consumption = [0]*(1+len(tram.heat_pumps))
    if heat_demand == 0:
        return electricity_consumption

    heat_demand_remainder = heat_demand
    for hp in tram.heat_pump_hierarchy:
        heat = min(heat_demand_remainder, tram.heat_pumps[hp])
        heat_demand_remainder -= heat
        electricity_consumption[1 + hp] = heat/tram.cop_pumps[hp]
    heat = min(heat_demand_remainder, tram.heat_resistive)
    heat_demand_remainder -= heat
    electricity_consumption[0] = heat

    if (heat_demand_remainder > 0):
        raise ExceededHeatingPowerError('Tram \'' + tram.name + '\' does not have enough heaters to satisfy the '
                                        'instantaneous heating demand.'
                                        'Please add heaters and/or increase their thermal heating power!')

    return electricity_consumption


# function calculate_consumption:
# parameters:
#   0, trams:               Array of class tram objects
#   1, climate_schedule:    Class climate_schedule object
#   2, T_tram:              Array of set-point temperatures [°C]
# returns: Instantaneous heat array [W] (0: total heating demand, 1: sun radiative heating, 2: passenger heating,
# 3: auxiliary device heating, 4: convective losses, 5: ventilation losses, 6: open door losses)

def calculate_consumption(trams, climate_schedule, T_tram):
    # sort set-point temperatures
    T_tram = sorted(T_tram)

    heat_total = [0] * len(T_tram)
    electricity_total = [0] * len(T_tram)

    # initialize arrays containing month, hour and tram name

    months = []
    for m in range(0, 12):
        if climate_schedule.operation_days[m] != 0:
            months.append(m)

    hours_tmp = []
    hours_crop = []
    for h in range(math.trunc(climate_schedule.t_begin), math.trunc(climate_schedule.t_end + 0.999999)):
        hmod = int(math.fmod(h, 24))
        if (hmod != h):
            hours_crop.append(hmod)
        else:
            hours_tmp.append(h)
    hours = [0]*(len(hours_crop) + len(hours_tmp))
    h = 0
    for hour in hours_crop:
        hours[h] = hour
        h += 1
    for hour in hours_tmp:
        hours[h] = hour
        h += 1

    tram_names = []
    for tram in trams:
        tram_names.append(tram.name)

    # dimension heat and electricity arrays

    heat_instantaneous = [0] * len(T_tram)
    electricity_instantaneous = [0] * len(T_tram)
    for T in range(0, len(T_tram)):
        heat_instantaneous[T] = [0] * len(months)
        electricity_instantaneous[T] = [0] * len(months)
        for m in range(0, len(months)):
            heat_instantaneous[T][m] = [0] * len(hours)
            electricity_instantaneous[T][m] = [0] * len(hours)
            for h in range(0, len(hours)):
                heat_instantaneous[T][m][h] = [0] * len(trams)
                electricity_instantaneous[T][m][h] = [0] * len(trams)

    # calculate heat and consumption
    for T in range(0, len(T_tram)):
        for m in range(0, len(months)):
            for h in range(0, len(hours)):
                hour_climate_data = int(math.fmod(hours[h]+TIME_SHIFT_TO_UTC, 24))
                h_delta = 1
                if (hours[h] == math.trunc(climate_schedule.t_begin)):
                    h_delta = 1 - (climate_schedule.t_begin - hours[h])
                if (hours[h] == int(math.fmod(math.trunc(climate_schedule.t_end), 24))):
                    h_delta = math.fmod(climate_schedule.t_end - hours[h], 24)
                for t in range(0, len(trams)):
                    A_abs = calculate_absorption_area(tram_length=trams[t].length, tram_width=trams[t].width,
                                                      tram_height=trams[t].height,
                                                      latitude=climate_schedule.latitude, month=months[m], hour=hours[h])
                    heat_instantaneous_tmp = calculate_heating_instantaneous(trams[t], T_tram[T],
                                                                            climate_schedule.T_environment[months[m]][hour_climate_data],
                                                                            climate_schedule.I_sun[months[m]][hour_climate_data],
                                                                            climate_schedule.f_sun, max(0, A_abs))
                    electricity_instantaneous_tmp = model_electricity_consumption(trams[t], heat_instantaneous_tmp[0])

                    heat_total[T] += (trams[t].tram_count * trams[t].tram_fraction_operational * heat_instantaneous_tmp[0]) * h_delta
                    electricity_total[T] += (tram.tram_count * tram.tram_fraction_operational * sum(electricity_instantaneous_tmp)) * h_delta

                    heat_instantaneous[T][m][h][t] = heat_instantaneous_tmp
                    electricity_instantaneous[T][m][h][t] = electricity_instantaneous_tmp

    electricity_cost_total = climate_schedule.cost_of_electricity * np.array(electricity_total)

    savings_total = []
    electricity_cost_total_max = electricity_cost_total[len(T_tram) - 1]
    for T in range(0, len(T_tram)):
        if (T == len(T_tram) - 1):
            savings_total.append(' ')
        else:
            percentage = 100 * (electricity_cost_total_max - electricity_cost_total[T]) / electricity_cost_total_max
            savings_total.append(str(round(percentage, 1)) + '%')

    return Consumption(heat_total, electricity_total, electricity_cost_total, savings_total,
                       T_tram, months, hours, tram_names, heat_instantaneous, electricity_instantaneous)


# function extract_data:
def extract_instantaneous_results(heat_instantaneous, electricity_instantaneous,
                               T_setpoint_temperatures, T_setpoint_temperature, months, month, hours, hour, tram_names, tram_name):
    T = T_setpoint_temperatures.index(T_setpoint_temperature)
    m = months.index(month)
    h = hours.index(hour)
    t = tram_names.index(tram_name)

    return [heat_instantaneous[T][m][h][t], electricity_instantaneous[T][m][h][t]]


# function print_total:
def print_total(consumption):
    print('----- TOTAL ENERGY CONSUMPTION RESULTS -----')
    print('T_setpoint=\t' + str(consumption.T_setpoint_temperatures) + '°C')
    print('Q_heat=\t\t' + str(np.array(consumption.heat_total) * 1e-6) + ' MWh')
    print('P_el=\t\t' + str(np.array(consumption.electricity_total) * 1e-6) + ' MWh')
    print('Cost=\t\t' + str(np.array(consumption.electricity_cost_total) * 1e-3) + ' kCHF')


# function print_instantaneous:
def print_instantaneous(consumption, T_setpoint, month, hour, tram_name):
    print('\n----- SELECTION [' + str(T_setpoint) + '°C, month ' + str(month+1) + ' at ' + str(hour) + ':00, tram \'' + tram_name + '\'] -----')
    results = extract_instantaneous_results(consumption.heat_instantaneous, consumption.electricity_instantaneous,
                                            consumption.T_setpoint_temperatures, T_setpoint,
                                            consumption.months, month,
                                            consumption.hours, hour,
                                            consumption.tram_names, tram_name)
    print('T_setpoint=\t' + str(T_setpoint) + '°C')
    print('Q_heat=\t\t' + str(np.array(results[0]) * 1e-3) + ' kW')
    print ('\t\t\t(0: total heating demand, 1: sun radiative heating, 2: passenger heating, 3: auxiliary device heating,'
           '4: convective losses, 5: ventilation losses, 6: open door losses)')
    print('P_el=\t\t' + str(np.array(results[1]) * 1e-3) + ' kW')

    #for t in range(0, len(consumption.tram_names)):
    #    print('\n----- INSTANTANEOUS POWER RESULTS FOR TRAM \'' + consumption.tram_names[t] + '\' -----')
    #    for i in range(0, len(consumption.month_instantaneous_per_tram[t])):
    #        print('[m+1=' + str(consumption.month_instantaneous_per_tram[t][i]+1) + ', '
    #              'T_tram=' + str(consumption.T_setpoint_instantaneous_per_tram[t][i]) + '°C]:\t\t' +
    #              'Q_heat=' + str(consumption.heat_instantaneous_per_tram[t][i][0]/1000) + ' kW,\t' +
    #              'P_el=' + str(sum(consumption.electricity_instantaneous_per_tram[t][i])/1000) + ' kW')


# function generate_plot_total:
def generate_plot_total(consumption):

    temperatures = ['']*len(consumption.T_setpoint_temperatures)
    for T in range(0, len(consumption.T_setpoint_temperatures)):
        temperatures[T] = str(round(consumption.T_setpoint_temperatures[T], 2)) + '°C'
    heat = np.array(consumption.heat_total) * 1e-6
    electricity = np.array(consumption.electricity_total) * 1e-6
    cost = np.array(consumption.electricity_cost_total) * 1e-6

    bar_width = 0.3
    index = np.arange(len(temperatures))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar1 = ax1.bar(index - bar_width, heat, width=bar_width, label='Power 1', color='darkred')
    bar2 = ax1.bar(index, electricity, width=bar_width, label='Power 2', color='orange')

    ax2 = ax1.twinx()
    bar3 = ax2.bar(index + bar_width, cost, width=bar_width, label='Cost', color='green')

    ax1.set_xlabel('Set-point temperature')
    ax1.set_ylabel('Energy [MWh]', color='black')
    ax2.set_ylabel('Cost [MCHF]', color='black')
    ax1.set_title('Total Energy and Cost')

    ax1.set_xticks(index)
    ax1.set_xticklabels(temperatures)

    ax1.grid(axis='y', linestyle='--', alpha=0.7, which='both', markevery=1)

    bars = [bar1, bar2, bar3]
    labels = ['Heat energy generation', 'Electric energy consumption', 'Cost for electricity']
    ax2.legend(bars, labels, loc='upper left')

    return fig
    


# function generate_plot_total:
def generate_plot_instantaneous(consumption, T_setpoint, month, hour):
    # data selection

    tram_count = len(consumption.tram_names)
    data_upper = []
    data_lower = []
    categories = [0]*2*len(consumption.tram_names)

    for t in range(tram_count-1, -1, -1):
        results = extract_instantaneous_results(consumption.heat_instantaneous, consumption.electricity_instantaneous,
                                                consumption.T_setpoint_temperatures, T_setpoint,
                                                consumption.months, month,
                                                consumption.hours, hour,
                                                consumption.tram_names, consumption.tram_names[t])
        #st.write(results)
        data_upper.append([results[0][0]*1e-3+0.000001, results[0][1]*1e-3+0.000001, results[0][2]*1e-3+0.000001, results[0][3]*1e-3+0.000001])
        data_lower.append([results[0][4]*1e-3+0.000001, results[0][5]*1e-3+0.000001, results[0][6]*1e-3+0.000001])
        categories[2*t] = consumption.tram_names[t] + ' (heating)'
        categories[2*(t+1)-1] = consumption.tram_names[t] + ' (losses)'

   # st.write(categories)
   # print("next")
   # st.write(data_lower)


    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.25)

    bars = []
    bar_height = 0.3
    y_upper = np.array(range(0, len(data_upper))) + 0.6*bar_height
    y_lower = np.array(range(0, len(data_upper))) - 0.6*bar_height
    labels_upper = ['Heaters', 'Solar heating', 'Heating by passengers', 'Auxiliary device heating']
    labels_lower = ['Convective losses', 'Ventilation losses', 'Open door losses']
    color_upper = ['darkred', 'orange', 'purple', 'gray']
    color_lower = ['darkblue', 'lightblue', 'skyblue']

    sum_upper = [0]*tram_count
    for i in range(0, len(labels_upper)):
        bar = ax.barh(y_upper, [data_upper[j][i] for j in range(tram_count)], left=sum_upper,
                       label=labels_upper[i], height=bar_height, color=[color_upper[i]])
        for t in range(0,tram_count):
            sum_upper[t] += data_upper[t][i]
        bars.append(bar)
    sum_lower = [0]*tram_count
    for i in range(0, len(labels_lower)):
        bar = ax.barh(y_lower, [data_lower[j][i] for j in range(tram_count)], left=sum_lower,
                      label=labels_lower[i], height=bar_height, color=[color_lower[i]])
        for t in range(0, tram_count):
            sum_lower[t] += data_lower[t][i]
        bars.append(bar)

    y_center = [0]*2*tram_count
    for t in range(0, tram_count):
        y_center[2*t] = y_upper[tram_count-t-1]
        y_center[2*(t+1)-1] = y_lower[tram_count-t-1]

    ax.set_xlabel('Power [kW]')
    ax.set_yticks(y_center)
    ax.set_yticklabels(categories)
    #ax.set_yticklabels(["D","C","B","A"])
    ax.minorticks_on()
    ax.set_title('Instantaneous heating and losses (for T_setpoint=' + str(T_setpoint) + '°C, month=' + str(month+1)+ ', time=' + str(hour) + ':00)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelrotation=0)
    ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    return fig


##### FRONT-END #####

st.set_page_config(layout="centered")

st.title("Energy Saving in Tram Heating - Calculator")



# Take inputs : - Length, width, height (in m)
# Average operation from [time] to [time] (in hours, minutes)
# Average tram count in operation
# Average instantaneous passenger number per tram
# Average fresh air supply via ventilation (in m³/h)
# Average fresh air supply from door openings (in m³/h)
# Average convection coefficient (in W/(m²K) )
# Button "Load VBZ values for Cobra tram" (or something)* """

# Defaults will be set to VBZ Cobra tram values

global trams
global T_tram
global heater_inputs
T_tram = []
trams = []
heater_inputs = [] 

#region Specifications

# Get number of tram types
tram_count = st.selectbox("Select Number of Tram Types", range(1,11))

tabs = {}

tab_names = [f"Tram {i+1}" for i in range(tram_count)]
tabs = st.tabs(tab_names)


def tram_inputs(i):

    with tabs[i]:
        # Function to set defaults for each input
        def defaults (i, value):
            if value == "name":
                st.session_state[f"name_{i}"] = default_values["Name"]
            elif value == "length":
                st.session_state[f"length_{i}"] = default_values["Length [m]"]
            elif value == "width":
                st.session_state[f"width_{i}"] = default_values["Width [m]"]
            elif value == "height":
                st.session_state[f"height_{i}"] = default_values["Height [m]"]
            elif value == "tram_count":
                st.session_state[f"tram_count_{i}"] = default_values["Tram count"]
            elif value == "passenger_count":
                st.session_state[f"passenger_count_{i}"] = default_values["Average passenger count"]
            elif value == "ventilation":
                st.session_state[f"ventilation_{i}"] = default_values["Average fresh air supply via ventilation [m³/h]"]
            elif value == "door_openings":
                st.session_state[f"door_openings_{i}"] = default_values["Average fresh air supply from door openings [m³/h]"]
            elif value == "convection":
                st.session_state[f"convection_{i}"] = default_values["Average convection coefficient [W/(m²K)]"]
            elif value == "tram_frac":
                st.session_state[f"tram_frac_{i}"] = default_values["Fraction of Trams in Operation"]
            elif value == "auxillary_heat":
                st.session_state[f"auxillary_heat_{i}"] = default_values["Auxillary Heat generated in tram [W]"]
            elif value == "all":
                st.session_state[f"name_{i}"] = default_values["Name"]
                st.session_state[f"length_{i}"] = default_values["Length [m]"]
                st.session_state[f"width_{i}"] = default_values["Width [m]"]
                st.session_state[f"height_{i}"] = default_values["Height [m]"]
                st.session_state[f"tram_count_{i}"] = default_values["Tram count"]
                st.session_state[f"passenger_count_{i}"] = default_values["Average passenger count"]
                st.session_state[f"ventilation_{i}"] = default_values["Average fresh air supply via ventilation [m³/h]"]
                st.session_state[f"door_openings_{i}"] = default_values["Average fresh air supply from door openings [m³/h]"]
                st.session_state[f"convection_{i}"] = default_values["Average convection coefficient [W/(m²K)]"]
                st.session_state[f"tram_frac_{i}"] = default_values["Fraction of Trams in Operation"]
                st.session_state[f"auxillary_heat_{i}"] = default_values["Auxillary Heat generated in tram [W]"]


        st.markdown(f" ## Tram Inputs {i+1}:")

        st.markdown("### Tram Specifications:")
    
        if "reset" not in st.session_state:
            st.session_state.reset = False

        with st.container():

            default_values = {
            "Name": "Cobra VBZ",
            "Length [m]": 25.9,
            "Width [m]": 2.4,
            "Height [m]": 3.6,
            "Tram count": 88,
            "Average passenger count": 37,
            "Average fresh air supply via ventilation [m³/h]": 1345,
            "Average fresh air supply from door openings [m³/h]": 3215,
            "Average convection coefficient [W/(m²K)]": 2.8,
            "Fraction of Trams in Operation": 0.9,
            "Auxillary Heat generated in tram [W]": 1500
            }

            col1, col2, col3 = st.columns([0.4, 0.2, 0.4])

            col1.button("Set all values to Cobra VBZ", on_click=defaults, args=[i, "all"], key=f"default_all_{i}")

            def reset(i):
                st.session_state[f'length_{i}'] = 0
                st.session_state[f'width_{i}'] = 0
                st.session_state[f'height_{i}'] = 0
                st.session_state[f'tram_count_{i}'] = 0
                st.session_state[f'passenger_count_{i}'] = 0
                st.session_state[f'ventilation_{i}'] = 0
                st.session_state[f'door_openings_{i}'] = 0
                st.session_state[f'convection_{i}'] = 0
                st.session_state[f'tram_frac_{i}'] = 0

            col2.button("Reset All Values", on_click=reset, args=[i], key=f"reset_{i}")

            
            name = st.text_input("Name", key=f"name_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "name"], key=f"default_name_{i}")
            length = st.number_input("Length [m]", key=f"length_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "length"], key=f"default_length_{i}")
            width = st.number_input("Width [m]", key=f"width_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "width"], key=f"default_width_{i}")
            height = st.number_input("Height [m]", key=f"height_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "height"], key=f"default_height_{i}")
            tram_count = st.number_input("Tram count", key=f"tram_count_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "tram_count"], key=f"default_tram_count_{i}")
            passenger_count = st.number_input("Average Passenger Count", key=f"passenger_count_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "passenger_count"], key=f"default_passenger_count_{i}")
            ventilation = st.number_input("Average fresh air supply via ventilation [m³/h]", key=f"ventilation_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "ventilation"], key=f"default_ventilation_{i}")
            door_openings = st.number_input("Average fresh air supply from door openings [m³/h]", key=f"door_openings_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "door_openings"], key=f"default_door_openings_{i}")
            convection = st.number_input("Average convection coefficient [W/(m²K)]", key=f"convection_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "convection"], key=f"default_convection_{i}")
            tram_frac = st.number_input("Fraction of Trams in Operation", key=f"tram_frac_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "tram_frac"], key=f"default_tram_frac_{i}")
            auxillary_heat = st.number_input("Auxillary Heat generated in tram [W]", key=f"auxillary_heat_{i}")
            st.button("Load Cobra VBZ value", on_click=defaults, args = [i, "auxillary_heat"], key=f"default_auxillary_heat_{i}")
        
        
        #endregion

        #region: Heater Inputs

        st.markdown(f" ### Heating Specifications for Tram {i+1}:")


        def heater_options(heater):

            st.write(f"Heater {heater + 1}")

            # Dropdown to select type of heater
            heater_type = st.selectbox(f"Select Heater Type {heater+1}", ["Heat Pump", "Resistance"], key = f"heatertype{i+1}_{heater+1}")

            # Maximum thermal power (input field)
            max_thermal_power = st.number_input("Maximum Thermal Power (kW):", min_value=1.0, value=100.0, format="%.1f", key = f"max_thermal_power{i+1}_{heater+1}")

            if heater_type == "Heat Pump":
                # Input field for average COP
                cop = st.number_input("COP at maximum power:", min_value=1.0, value=3.0, format="%.2f", key = f"cop{i+1}_{heater+1}")
            else:
                cop = "Not Applicable"

            heater_dict = ({
                "Heater Type": heater_type,
                "Maximum Thermal Power (kW)": max_thermal_power,
                "COP (full load)": cop,
                "Tram": f"Tram_{i+1}"
            })

            heater_inputs.append(heater_dict)



        if f'heaters_{i}' not in st.session_state:
            st.session_state[f'heaters_{i}'] = 0

        def add_heater_option():
            st.session_state[f'heaters_{i}'] += 1

        def reset_heaters():
            st.session_state[f'heaters_{i}'] = 0

        if f'heaters_{i}' in st.session_state:
            for heater in range(st.session_state[f'heaters_{i}']):
                heater_options(heater)

        st.button("Add Heater", on_click=add_heater_option, key=f"add_heater_{i}")

        # Create a button to reset the input section
        reset_button = st.button("Remove Heaters", on_click=reset_heaters, key=f"reset_heaters_{i}")

        # Print the user inputs
        
        df = pd.DataFrame(heater_inputs)

        if not df.empty: 
            
            df = df[df['Tram'] == f"Tram_{i+1}"]
            resistive_df = df[df['Heater Type'] == 'Resistance']
            if not resistive_df.empty:
                resistive_heat = np.array(df[df['Heater Type'] == 'Resistance']['Maximum Thermal Power (kW)'].sum())*1e3
            else:
                resistive_heat = 0
            heat_pump_df = df[df['Heater Type'] == 'Heat Pump']
            if not heat_pump_df.empty:
                heat_pumps = np.array(df[df['Heater Type'] == 'Heat Pump']['Maximum Thermal Power (kW)'].values)*1e3
                cops = df[df['Heater Type'] == 'Heat Pump']['COP (full load)'].values
            else:
                heat_pumps = []
                cops = []
        else:
            resistive_heat = 0
            heat_pumps = []
            cops = []

        #endregion

        new_tram = Tram(name=name,passenger_number=passenger_count,volume_flow_ventilation=ventilation/3600,
                        volume_flow_doors=door_openings/3600,length=length,width=width,height=height,k_chassis=convection,
                        heat_auxiliary=auxillary_heat,heat_resistive=resistive_heat,heat_pumps=heat_pumps,cop_pumps=cops,
                        tram_count=tram_count,tram_fraction_operational=tram_frac)
        trams.append(new_tram)
if "tram_count" not in st.session_state:
    st.session_state["tram_count"] = tram_count

for i in range (tram_count):
    tram_inputs(i)

for i in range(tram_count):
    with tabs[i]:
        selected_heaters = [heater for heater in heater_inputs if heater["Tram"] == f"Tram_{i+1}"]
        df = pd.DataFrame(selected_heaters)
        if not df.empty:
            df = df.drop(columns=["Tram"])
            st.dataframe(df, hide_index=True)
        else:
            st.write("No heaters added yet.")
          

#region: Temperature Inputs

st.markdown(" ## Temperature Settings:")

T_tram = []

if 'temp_inputs' not in st.session_state:
    st.session_state['temp_inputs'] = 0

def temp_options(temp):
    
        st.write(f"Set point temperature {temp + 1}")
    
        # Maximum thermal power (input field)
        temp_value = st.number_input("Set Point Temperature [°C]:", min_value=5.0, max_value=30.0, step = 0.5, value=18.0, format="%.1f", key = f"temp_value_{temp}_{i}")
    
        T_tram.append(temp_value)
        #st.write(T_tram)

def add_temp_option():
    st.session_state['temp_inputs'] += 1


def reset_temp():
    st.session_state['temp_inputs'] = 0

if 'temp_inputs' in st.session_state:
    for temp in range(st.session_state['temp_inputs']):
        temp_options(temp)

st.button("Add Temperature Setting", on_click=add_temp_option, key=f"add_temp_{i}")
st.button("Reset Temperature Settings", on_click=reset_temp, key=f"reset_temp_{i}")





# endregion

# region operating condition

st.markdown("## Operating Schedule for all Trams:")

# Get operating schedule

location = st.selectbox('Select City', LIST_OF_CITIES)
begin_month = st.number_input("Month of the year when operation of trams is started (1-12)", min_value=1, max_value = 12, value=1, step=1, key=f"begin_month_{i}")
begin_day = st.number_input("Day of the month when operation of trams is started (1-31):", min_value=1, max_value = 31, value=1, step=1, key=f"begin_day_{i}")
end_month = st.number_input("Month of the year when operation of trams is ended (1-12)", min_value=1, max_value = 12, value=12, step=1, key=f"end_month_{i}")
end_day = st.number_input("Day of the month when operation of trams is ended (1-31):", min_value=1, max_value = 31, value=31, step=1, key=f"end_day_{i}")
begin_hour = st.number_input("Hour of starting operation every day (0-24)", min_value=0, max_value=24, value=5,step = 1, key=f"begin_hour_{i}")
end_hour = st.number_input("Hour of ending operation every day (0-24)", min_value=0, max_value=24, step = 1, value=24, key=f"end_hour_{i}")

#endregion

#region Electricity Costs

st.markdown("## Electricity Costs:")

# Get cost of electricity
cost_electricity = st.number_input("Cost of electricity [CHF/kWh]:", min_value=0.01, value=0.2, format="%.2f", key = "cost_electricity")

#endregion


# region calculate consumption
# Button which executes Python function to calculate and display results

def generate_results():
    T_location = load_temperature_data(location)
    I_sun_zurich = load_solar_irradiation_data(location)
    climate_schedule = ClimateSchedule(begin_month, begin_day, end_month, end_day, begin_hour, end_hour,
                                       47.375, T_location, I_sun_zurich, 0.8, cost_electricity)
    

    # calculation
    consumption = calculate_consumption(trams, climate_schedule, T_tram)
    return consumption, climate_schedule

calc_button = st.button("Calculate", key="calculate")

consumption = Consumption([],[], [], [], [], [], [], [], [], [])

if "consumption" not in st.session_state:
    st.session_state["consumption"] = consumption

if calc_button:
    #st.write(T_tram)
    if len(heater_inputs) < tram_count or T_tram == []:
    
        if len(heater_inputs) < tram_count:
            st.markdown(
                '<span style="color: red; font-size: 20px;">Please add heater(s) for every tram before calculating!</span>',
                unsafe_allow_html=True
            )
        elif T_tram == []:
            st.markdown(
                '<span style="color: red; font-size: 20px;">Please add set-point temperature(s) before calculating!</span>',
                unsafe_allow_html=True
            )

    else:

        try:
            # Dedup logic for list of temperatures
            T_tram = list(dict.fromkeys(T_tram))
            consumption, climate_schedule_final = generate_results()
            heat_inst = consumption.heat_instantaneous
            index_tuples = []
            values = []

            def get_output_csv(heat_inst = [], elec_inst = []):

                def flatten_list(nested_list,current_index):
                    for i, value in enumerate(nested_list):
                        if isinstance(value, list):
                            flatten_list(value,current_index + [i])
                        else:
                            index_tuples.append(current_index + [i])
                            values.append(value)
                
                flatten_list(heat_inst, [])

                df_heat = pd.DataFrame(values, index=pd.MultiIndex.from_tuples(index_tuples, names=['Temperature', 'Month', 'Hour', 'Tram', 'Heat_Type']), columns=['Value'])
                heat_type_mapper = {0: 'Solar heating', 1: 'Passenger Heat', 2: 'Auxillary Heat', 3: 'Convective Losses', 4: 'Ventilation Losses', 5: 'Open Door Losses', 6: 'Heating Demand'} 
                tram_mapper = {}
                for i, tram in enumerate(trams):
                    tram_mapper[i] = tram.name
                temp_mapper = {}
                for i, temp in enumerate(T_tram):
                    temp_mapper[i] = temp
                df_heat = df_heat.reset_index()
                df_heat = df_heat.replace({"Temperature": temp_mapper})
                df_heat = df_heat.replace({"Heat_Type": heat_type_mapper})
                df_heat = df_heat.replace({"Tram": tram_mapper})

                df_elec = pd.DataFrame()

                return df_heat, df_elec

            df_heat, df_elec = get_output_csv(heat_inst)
            
            if "df_heat" not in st.session_state:
                st.session_state["df_heat"] = df_heat
            st.session_state["consumption"] = consumption
            if "climate_schedule" not in st.session_state:
                st.session_state["climate_schedule"] = climate_schedule_final

            st.markdown("## Total Energy and Cost:")

            ec_data = []
            for T in range(0, len(consumption.T_setpoint_temperatures)):
                ec_data.append([round(consumption.T_setpoint_temperatures[T], 2),
                                round(consumption.heat_total[T] * 1e-6, 2),
                                round(consumption.electricity_total[T] * 1e-6, 2),
                                round(consumption.electricity_cost_total[T] * 1e-6, 2),
                                consumption.savings_total[T]])
            ec_df = pd.DataFrame(ec_data, columns=("T_setpoint[°C]", "Heat energy [MWh]", "Electricity consumption [MWh]", "Cost [MCHF]", "Savings (cmp. to T=" + str(round(consumption.T_setpoint_temperatures[len(consumption.T_setpoint_temperatures)-1], 2)) + "°C)"))
            st.dataframe(ec_df, hide_index=True, column_config= {'T_setpoint<br>[°C]':st.column_config.NumberColumn(format="%.2f"), "Heat energy\n[MWh]":st.column_config.NumberColumn(format="%.2f"), "Electricity consumption [MWh]":st.column_config.NumberColumn(format="%.2f"), "Cost [MCHF]":st.column_config.NumberColumn(format="%.2f"), "Savings (cmp. to T=" + str(round(consumption.T_setpoint_temperatures[len(consumption.T_setpoint_temperatures)-1], 2)) + "°C)":st.column_config.TextColumn()})
            if "ec" not in st.session_state["ec"]:
                st.session_state["ec"] = ec_df
            st.write(generate_plot_total(consumption))
            st.session_state['calc_complete'] = True
            #st.write(consumption)
        except NameError:
            st.markdown(
                '<span style="color: red; font-size: 20px;">Please Enter All Inputs!</span>',
                unsafe_allow_html=True
            )
        except ExceededHeatingPowerError as e:
            st.markdown(
                '<span style="color: red; font-size: 20px;">' + e.message + '</span>',
                unsafe_allow_html=True
            )

# endregion

# region display results

st.markdown("## Instantaneous Power Values per Tram (available after calculation):")
inst_button_hide = True
if "calc_complete" in st.session_state:
    if st.session_state['calc_complete']:
        inst_button_hide = False
    else:
        inst_button_hide = True

if "ec" not in st.session_state:
    st.session_state["ec"] = pd.DataFrame()

if "calc_complete" in st.session_state:
    if not st.session_state['calc_complete']:
        pass
    else:
        consumption = st.session_state["consumption"]
        #st.write(consumption)
        ins_T_setpoint = st.selectbox("Select Set Point Temperature", consumption.T_setpoint_temperatures)
        month_list = [month + 1 for month in consumption.months]
        ins_month = st.selectbox("Select Month", month_list) - 1
        ins_hour = st.selectbox("Select Hour", consumption.hours)
        # Save to dataframe 
        T_environment = st.session_state["climate_schedule"].T_environment[ins_month][ins_hour]
        df_operating = pd.DataFrame({"Outside Temperature °C":T_environment,"Setpoint Temperature":ins_T_setpoint, "Month":ins_month+1, "Hour":ins_hour}, index=[0])
        if "df_operating" not in st.session_state:
            st.session_state["df_operating"] = df_operating

inst_button = st.button("Show Instantaneous Power Values", key="show_instantaneous_power_values",disabled=inst_button_hide)

if inst_button:
    ins_data = []
    for T in range(0, tram_count):
        results = extract_instantaneous_results(consumption.heat_instantaneous, consumption.electricity_instantaneous,
                                                consumption.T_setpoint_temperatures, ins_T_setpoint,
                                                consumption.months, ins_month,
                                                consumption.hours, ins_hour,
                                                consumption.tram_names, consumption.tram_names[T])
        ins_data.append([consumption.tram_names[T],
                         round(results[0][0]*1e-3, 2),
                         round(results[0][1]*1e-3, 2),
                         round(results[0][2]*1e-3, 2),
                         round(results[0][3] * 1e-3, 2),
                         round(results[0][4] * 1e-3, 2),
                         round(results[0][5] * 1e-3, 2),
                         round(results[0][6] * 1e-3, 2),
                         round(np.sum(results[1])*1e-3, 2)])
    ins = pd.DataFrame(ins_data, columns=(
        ['Tram', 'Heating power [kW]', 'Solar heat [kW]', 'Passenger heat [kW]', 'Aux. device heat [kW]',
         'Convective losses [kW]', 'Ventilation losses [kW]', 'Open door losses [kW]', 'Electricity consumption [kW]']))
    st.divider()
    st.markdown(" ## Results: ")
    st.markdown(" ### Total Energy and Costs: ")
    ec_df = st.session_state["ec"]
    st.dataframe(ec_df, hide_index=True, column_config= {'T_setpoint [°C]':st.column_config.NumberColumn(format="%.2f"), "Heat energy [MWh]":st.column_config.NumberColumn(format="%.2f"), "Electricity consumption [MWh]":st.column_config.NumberColumn(format="%.2f"), "Cost [MCHF]":st.column_config.NumberColumn(format="%.2f"), "Savings (cmp. to T=" + str(round(consumption.T_setpoint_temperatures[len(consumption.T_setpoint_temperatures)-1], 2)) + "°C)":st.column_config.NumberColumn(format="%.2f")})
    st.write(generate_plot_total(consumption))
    st.divider()
    st.markdown(" ### Selected Instantaneous Period:")
    df_operating = st.session_state["df_operating"]
    st.dataframe(df_operating, hide_index=True, column_config= {'Outside Temperature °C': st.column_config.NumberColumn(format="%.2f"),'Setpoint Temperature':st.column_config.NumberColumn(format="%.2f"), "Month":st.column_config.NumberColumn(format="%.0f"), "Hour":st.column_config.NumberColumn(format="%.0f")})
    st.markdown(" ### Instantaneous Power Values: ")
    st.dataframe(ins,hide_index= True, column_config= {'Tram':st.column_config.TextColumn(), 'Heating power [kW]':st.column_config.NumberColumn(format="%.2f"), 'Solar heat [kW]':st.column_config.NumberColumn(format="%.2f"), 'Passenger heat [kW]':st.column_config.NumberColumn(format="%.2f"), 'Aux. device heat [kW]':st.column_config.NumberColumn(format="%.2f"), 'Convective losses [kW]':st.column_config.NumberColumn(format="%.2f"), 'Ventilation losses [kW]':st.column_config.NumberColumn(format="%.2f"), 'Open door losses [kW]':st.column_config.NumberColumn(format="%.2f"), 'Electricity consumption [kW]':st.column_config.NumberColumn(format="%.2f")})
    st.write(generate_plot_instantaneous(consumption, ins_T_setpoint, ins_month,  ins_hour))
    df_heat = st.session_state["df_heat"]
    csv_heat = df_heat.to_csv(index=True).encode("utf-8")
    st.download_button("Download Instantaneous Heat Data",csv_heat, "Instantaneous_Heat.csv", mime="text/csv")


disclaimer_text = """
**Disclaimer:**
Responsible for the content: Florian Schubert, Clara Tillous Oliva, Beatriz Movido, Oleksandr Halipchak, and Yash Dubey, students at ETH Zurich, Rämistrasse 101, 8092 Zürich, Switzerland (November 2023). All results and information are to be understood as estimates. No liability is taken for their accuracy or for the consequences of their use. Default tram data is used with permission from Verkehesbetriebe Zürich (VBZ). Climate data obtained from PVGIS EU.
"""
st.markdown(disclaimer_text)

# Add custom HTML and JavaScript for footer
footer_html = """
    <div id="footer" style="display: none; position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f1f1f1; padding: 10px; text-align: center;">
        <p>Your Footer Content Here</p>
    </div>
"""

# Add the custom HTML to the Streamlit app
st.markdown(footer_html, unsafe_allow_html=True)

# Add JavaScript to show/hide the footer based on scroll position
footer_js = """
    <script>
        window.onscroll = function() {
            var footer = document.getElementById("footer");
            if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
                footer.style.display = "block";
            } else {
                footer.style.display = "none";
            }
        };
    </script>
"""

# Add the JavaScript to the Streamlit app
st.markdown(footer_js, unsafe_allow_html=True)

# endregion