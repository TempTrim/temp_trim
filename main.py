# ETH Zurich, Energy Now 2.0, Energy Savings Modeling Tool, Thermodynamics
# Florian Schubert, 2023-11-13


# IMPORTS

from dataclasses import dataclass
import numpy as np



# CONSTANTS

C_P_AIR = 1005 # mass-specific heat capacity of air (at T=, p=, ...) [J/kgK]
RHO_AIR = 1.205 # density of air (at T=, p=, ...) [kg/m³]
HEAT_PERSON = 100 # average heat generation by on person (under ... conditions) [W]
# TODO specify C_P_AIR, RHO_AIR, HEAT_PERSON
MONTH_DAYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366] # day in year of monthly beginning



# FUNCTIONS FOR CLASSES

# function calc_area_convection
# parameters: tram dimensions [m]
# returns total tram effective convection area [m²]
def calc_area_convection(length, width, height):
    return 2*(length*height + length*width + width*height)

# function calc_area_absorption
# parameters: tram dimensions [m]
# returns (normal) effective area for radiative heat absorption from sun radiation [m²]
def calc_area_absorption(length, width, height):
    # TODO implement normal area
    return 0.5*(length * height + width * height)

# function volume_to_mass_flow
# parameter: volume air flow [m³/s]
# returns mass air flow [kg/s]
def volume_to_mass_flow(volume_flow):
    return RHO_AIR*volume_flow

# function calculate_sun_intensity
# parameter: daily radiative sun energy [Wh/m²] and sun / operational times [h]
# returns average instantaneous sun intensity [W/m²]
def calculate_sun_intensity(E_sun, t_sun_rise, t_sun_set, t_operation_begin, t_operation_end):
    t_sun_operation = min(t_sun_set, t_operation_end) - max(t_sun_rise, t_operation_begin)
    t_sun = t_sun_set - t_sun_rise
    t_operation = t_operation_end - t_operation_begin
    E_operation = E_sun * (t_sun_operation/t_sun)
    E_average = E_operation/t_operation
    return E_average

# function generate_heat_pump_hierarchy
# parameter: array of coefficient of performance for heat pumps (one entry per heat pump) [-]
# returns list with heat pump indices sorted by decreasing COP
def generate_heat_pump_hierarchy(cops):
    return (len(cops) - 1 - np.argsort(cops))



# function calculate_operation_days_per_month
# parameter: begin and end day (numbers in year)
# returns list with operation days per months (1 to 12 for January to December with indices 0 to 11)
def calculate_operation_days_per_month(day_begin, day_end):
    operation_days = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for m in range(0, 12):
        if day_begin <= day_end:
            day_begin_m = max(day_begin - 1, MONTH_DAYS[m])
            day_end_m = min(day_end, MONTH_DAYS[m + 1] - 1)
            operation_days[m] = max(0, day_end_m - day_begin_m + 1)
        else:
            day_begin_m = max(day_end, MONTH_DAYS[m])
            day_end_m = min(day_begin - 1, MONTH_DAYS[m + 1] - 1)
            operation_days[m] = (MONTH_DAYS[m + 1] - MONTH_DAYS[m]) - max(0, day_end_m - day_begin_m + 1)
    return operation_days



# CLASSES

# tram data class
@dataclass
class Tram:
    # class init function
    # parameters: see comments in function, except for
    #   - tram dimensions [m] length, width, height instead of effective areas
    #   - volume flows [m³/s] instead of mass flows
    def __init__(self, name, passenger_number, volume_flow_ventilation, volume_flow_doors, length, width, height,
                 k_chassis, absorptivity, heat_auxiliary, heat_resistive, heat_pumps, cop_pumps, tram_count,
                 tram_fraction_operational):
        # single vehicle specification
        self.name = name # name
        self.passenger_number = passenger_number # total average passenger number per tram [-]
        self.mass_flow_ventilation = volume_to_mass_flow(volume_flow_ventilation) # total average mass flow by ventilation system [kg/s]
        self.mass_flow_doors = volume_to_mass_flow(volume_flow_doors) # total average mass flow by open doors [kg/s]
        self.area_convection = calc_area_convection(length, width, height) # effective area for convective heat losses [m²]
        self.area_absorption_sun = calc_area_absorption(length, width, height) # (normal) effective area for radiative heat absorption from sun radiation [m²]
        self.k_chassis = k_chassis # effective average heat transfer coefficient for convective losses [W/m²K]
        self.absorptivity = absorptivity # effective absorptivity for sun radiation fraction) [-]
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
    #   - day_begin and day_end as the day-number in the year that determine begin and end of the investigated period
    #   - daily average normal sun irradiative energy [Wh/m²] instead of average sun intensity
    def __init__(self, day_begin, day_end, t_begin, t_end, t_sun_rise, t_sun_set, T_environment, E_sun_daily, f_sun,
                 cost_of_electricity):
        # list with operation days per months (1 to 12 for January to December with indices 0 to 11)
        self.operation_days = calculate_operation_days_per_month(day_begin, day_end)

        self.t_begin = t_begin # daily average operation begin time [h]
        if t_end < t_begin:
            self.t_end = t_end + 24
        else:
            self.t_end = t_end
        # daily average operation end time [h]

        self.T_environment = T_environment # array of (12) monthly average environment temperatures [°C]
        self.I_sun = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for m in range(0, 12):
            self.I_sun[m] = calculate_sun_intensity(E_sun_daily[m], t_sun_rise[m], t_sun_set[m], t_begin, t_end)
        # array of (12) monthly average normal sun intensity [W/m²] normalized over operational hours (see function)

        self.f_sun = f_sun # yearly average un-shaded sun fraction [-]

        self.cost_of_electricity = cost_of_electricity # cost of electricity [CHF/Wh]



# consumption data class
@dataclass
class Consumption:
    # class init function
    # parameters: see comments in function
    def __init__(self, heat_total, electricity_total, electricity_cost_total,
                 tram_names, month_instantaneous_per_tram, T_setpoint_instantaneous_per_tram,
                 heat_instantaneous_per_tram, electricity_instantaneous_per_tram):
        self.heat_total = heat_total # total heat energy by heaters of all trams during whole interval [Wh]
        self.electricity_total = electricity_total # total electricity consumption by heaters of all trams during whole interval [Wh]
        self.electricity_cost_total = electricity_cost_total # total cost of electricity consumption by heaters of all trams during whole interval [CHF]

        self.tram_names = tram_names # array of tram names (in order of calculation)

        self.month_instantaneous_per_tram = month_instantaneous_per_tram # two dimensional array for month used for calculations
        self.T_setpoint_instantaneous_per_tram = T_setpoint_instantaneous_per_tram # two dimensional array for set-point temperatures used for calculations
        # first dimension: tram type, second dimension: calculation entry

        self.heat_instantaneous_per_tram = heat_instantaneous_per_tram # three dimensional array with instantaneous heat flows
        # first dimensional: tram type, second dimension: calculation entry,
        # third dimension: heat values [W] (0: total heating demand, 1: sun radiative heating, 2: passenger heating,
        #               3: auxiliary device heating, 4: convective losses, 5: ventilation losses, 6: open door losses)

        self.electricity_instantaneous_per_tram = electricity_instantaneous_per_tram # three dimensional array with instantaneous electricity consumptions
        # first dimensional: tram type, second dimension: calculation entry,
        # third dimension: electricity demand per device [W] (first entry resistive heater power,
        #                       electric heat pump powers following)



# FUNCTIONS

# function calculate_heating_instantaneous:
#   thermodynamical model of instantaneous heat generation in one tram
# parameters:
#   0, tram:            Class tram object
#   1, T_tram:          Set-point temperature for tram cabin [°C]
#   2, T_environment:   Temperature or environment [°C]
#   3, I_sun:           Average (normal) sun intensity [W/m²]
#   4, f_sun:           Sun fraction [-]
# returns: Instantaneous heat array [W] (0: total heating demand, 1: sun radiative heating, 2: passenger heating,
# 3: auxiliary device heating, 4: convective losses, 5: ventilation losses, 6: open door losses)
def calculate_heating_instantaneous(tram, T_tram, T_environment, I_sun, f_sun):
    heat_sun = tram.area_absorption_sun * I_sun * tram.absorptivity * f_sun
    heat_passenger = tram.passenger_number * HEAT_PERSON
    heat_auxiliary_devices = tram.heat_auxiliary
    heat_loss_convection = tram.area_convection * tram.k_chassis * (T_tram - T_environment)
    heat_loss_ventilation =  tram.mass_flow_ventilation * C_P_AIR * (T_tram - T_environment)
    heat_loss_doors =  tram.mass_flow_doors * C_P_AIR * (T_tram - T_environment)
    heating_demand = max(0, ((heat_loss_convection + heat_loss_ventilation + heat_loss_doors)
                      - (heat_sun + heat_passenger + heat_auxiliary_devices)))
    # todo sun heat transfer too high?
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

    return electricity_consumption



# function calculate_consumption:
# parameters:
#   0, trams:               Array of class tram objects
#   1, climate_schedule:    Class climate_schedule object
#   2, T_tram:              Array of set-point temperatures [°C]
# returns: Instantaneous heat array [W] (0: total heating demand, 1: sun radiative heating, 2: passenger heating,
# 3: auxiliary device heating, 4: convective losses, 5: ventilation losses, 6: open door losses)
def calculate_consumption(trams, climate_schedule, T_tram):

    tram_name = []
    month_instantaneous = []
    T_setpoint_instantaneous = []
    heat_instantaneous_per_tram = []
    electricity_instantaneous_per_tram = []

    heat_total = 0
    electricity_total = 0

    for tram in trams:
        tram_name.append(tram.name)
        month_instantaneous_tram = []
        T_setpoint_instantaneous_tram = []
        heat_instantaneous_tram = []
        electricity_instantaneous_tram = []
        for m in range(0,12):
            if climate_schedule.operation_days[m] != 0:
                for T_setpoint in T_tram:
                    #print('-----  m+1=' + str(m+1) + ', T=' + str(T_setpoint) + '  -----')
                    heat_instantaneous = calculate_heating_instantaneous(tram, T_setpoint, climate_schedule.T_environment[m],
                                                           climate_schedule.I_sun[m], climate_schedule.f_sun)
                    electricity_instantaneous = model_electricity_consumption(tram, heat_instantaneous[0])

                    month_instantaneous_tram.append(m)
                    T_setpoint_instantaneous_tram.append(T_setpoint)
                    heat_instantaneous_tram.append(heat_instantaneous)
                    electricity_instantaneous_tram.append(electricity_instantaneous)
                    #print('Q=' + str(heat_instantaneous[0]/1000) + ' kW,\tP_el=' + str(sum(electricity_instantaneous)/1000) + ' kW')

                    heat_total += (tram.tram_count * tram.tram_fraction_operational
                                   * (climate_schedule.t_end-climate_schedule.t_begin) * heat_instantaneous[0])

                    electricity_total += (tram.tram_count * tram.tram_fraction_operational
                                   * (climate_schedule.t_end - climate_schedule.t_begin) * sum(electricity_instantaneous))

        month_instantaneous.append(month_instantaneous_tram)
        T_setpoint_instantaneous.append(T_setpoint_instantaneous_tram)
        heat_instantaneous_per_tram.append(heat_instantaneous_tram)
        electricity_instantaneous_per_tram.append(electricity_instantaneous_tram)

    electricity_cost_total = climate_schedule.cost_of_electricity * electricity_total

    return Consumption(heat_total, electricity_total, electricity_cost_total,
                       tram_name, month_instantaneous, T_setpoint_instantaneous,
                       heat_instantaneous_per_tram, electricity_instantaneous_per_tram)



# function print_instantaneous:
def print_instantaneous(consumption):
    print('----- TOTAL ENERGY CONSUMPTION RESULTS -----')
    print('Q_heat=' + str(consumption.heat_total / 10e6) + ' MWh')
    print('P_el=' + str(consumption.electricity_total / 10e6) + ' MWh')
    print('Cost=' + str(consumption.electricity_cost_total / 10e3) + ' kCHF')

    for t in range(0, len(consumption.tram_names)):
        print('\n----- INSTANTANEOUS POWER RESULTS FOR TRAM \'' + consumption.tram_names[t] + '\' -----')
        for i in range(0, len(consumption.month_instantaneous_per_tram[t])):
            print('[m+1=' + str(consumption.month_instantaneous_per_tram[t][i]+1) + ', '
                  'T_tram=' + str(consumption.T_setpoint_instantaneous_per_tram[t][i]) + '°C]:\t\t' +
                  'Q_heat=' + str(consumption.heat_instantaneous_per_tram[t][i][0]/1000) + ' kW,\t' +
                  'P_el=' + str(sum(consumption.electricity_instantaneous_per_tram[t][i])/1000) + ' kW')





# TODO TEST

# definition
trams = [Tram(passenger_number=37, tram_count=30, tram_fraction_operational=0.9, heat_pumps=[5000, 4000, 3000], heat_auxiliary=1500,
             heat_resistive=102300, volume_flow_doors=0.89, volume_flow_ventilation=0.37, cop_pumps=[3.5, 4, 4.5], k_chassis=2.8,
             height=3.6, absorptivity=1.0, length=25.9, width=2.4, name='test tram')]
climate_schedule = ClimateSchedule(day_begin=74, day_end=161, t_begin=5, t_end=1, t_sun_set=[19]*12, t_sun_rise=[7]*12,
                                   E_sun_daily=[1000]*12, f_sun=0.8, T_environment=[-3, -1, 2, 5, 12, 18, 23, 18, 12, 5, 2, -1], cost_of_electricity=0.20)

# calculation
consumption = calculate_consumption(trams, climate_schedule, T_tram=[15, 17.5, 20])

# output
print_instantaneous(consumption)
