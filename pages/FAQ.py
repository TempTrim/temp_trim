import streamlit as st

st.title("FAQ")

about = st.expander("About", expanded=False)

with about:
    st.markdown("#### What is this app?")
    st.markdown("""
                This app is a thermodynamic model that models the savings 
                in energy consumption in trams by varying the setpoint temperature
                of the tram's heating system. The app is designed to be a tool used
                by tram operators to estimate potential energy savings and generate
                public support for eco-friendly intiatives. The project received the Audience Award at the Energy Now! 2.0 finale.
                """)
    st.divider()
    st.markdown("#### How does it work?")
    st.markdown("""
                The app uses a thermodynamic model to estimate the energy consumption
                of a tram. The model is based on the energy balance of the tram and
                the heat transfer between the tram and its surroundings. The model
                is validated on data provided by Verkehrsbetriebe Zürich (VBZ).

                The model used is described in a paper found [here](https://github.com/TempTrim/temp_trim/blob/main/paper_modelling.pdf).
                """)
    st.divider()
    st.markdown(" #### Who made this app?")
    st.markdown("""
                
                This app was developed by a team of students from ETH Zürich as a
                part of the Energy Now! 2.0 challenge. The team members are:
                - [Florian Schubert](https://linkedin.com/in/f-schubert)
                - [Clara Tillous Oliva](https://www.linkedin.com/in/clara-tillous-oliva-1089bb213/)
                - [Beatriz Movido](https://www.linkedin.com/in/bmovido/)
                - [Yash Dubey](https://www.linkedin.com/in/yashdubey132/)
                
                """)
    
how_to = st.expander("Usage", expanded=False)

with how_to:
    st.markdown("#### How do I use this app?")
    st.markdown(""" 

                To use this app, you need the following:

                - **Parameters or Specifications of the tram design:** These are usually available with the manufacturer of the tram.
                - **Heating Facilities in the tram:** These are details of all the heaters in the tram, including the type of heater, power, and COP (if the heater is a heat pump).
                - **Operating Schedule of the tram:** This is the schedule of operation for a typical tram in a year.
                - **Electricity Costs:** The cost of electricity for running the tram.

                All of these details must be included for all different types of trams that you want to evaluate. 
                For convenience, default values are provided for all parameters.
                """)
    st.divider()
    st.markdown("#### How do I interpret the results?")
    st.markdown(""" 
                The results are presented in two ways: 
                - **Overall Results**: A table provides the annual energy consumption
                and associated costs for each setpoint temperature for the entire network
                (all trams). The table provides an overview of percentage savings in energy
                consumption for each setpoint temperature, compared to the highest setpoint
                temperature. A graph is also provided to visualize the results.

                - **Instantaneous Results**: NOTE: This is only available after completing the calculations by clicking the "Calculate" button.
                You can select a specific temperature, month, day and hour
                to obtain instantaneous values for heating generated in the tram by sources
                such as solar radiation, passengers, and the tram's heating system. These results
                are also available in a graph and can be downloaded as a CSV file for further analysis.    
                 """)

errors = st.expander("Troubleshooting", expanded=False)

with errors:
    st.markdown("#### I am getting an error. What should I do?")
    st.markdown(""" 
                If you see errors in the app, please check the following: 
                - All parameters must be entered. If you do not have the value for a parameter,
                you can use the default value provided.
                - All trams must have a unique name.
                - All trams must have at least one heater with enough power to meet the heating demand.
                - There must be at least one set point temperature. For meaningful results, it is recommended
                to have at least two set point temperatures.

                If you still see errors, please refresh the page and try again. If the error persists,
                please contact the app developers by creating an issue on the [GitHub repository](https://github.com/TempTrim/temp_trim). 
                Include screenshots of the error to help us identify the problem.
                """)
contribute = st.expander("Contribute", expanded=False)

with contribute:
    st.markdown("#### How can I contribute to this app?")
    st.markdown(""" 
                We would love to hear from you! If you have any suggestions for improving the app,
                please create an issue on the [GitHub repository](https://github.com/TempTrim/temp_trim).
                If you'd like to be added as a contributor, please contact:
                """)

