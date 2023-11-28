# TempTrim - App to Calculate Tram Heat Savings

## Description
The app is designed to calculate savings in energy consumption in the form of heating demand for trams depending on setpoint temperatures. The app uses a thermodynamic model developed using experimental data from Verkehersbetreibe Zurich (VBZ). For more details about the model, refer to the [paper]()
The app is developed and hosted using [Streamlit](https://www.streamlit.io/).

## Table of Contents
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)
- [Contributing](#contributing)

## Usage
The app can be accessed using the following link: [TempTrim](https://temptrim.streamlit.app/). Details of inputs and interpretations of the results can be found on the [FAQ page](https://temptrim.streamlit.app/FAQ) of the app.

There is no cost associated with using the app. However, the app is hosted on a free server and hence, it is generally recommended to avoid making multiple requests at the same time. If you are interested in using the app for a large number of simulations, please contact the authors, or clone the repository and run the app locally.

## Requirements
The app currently runs on Python 3.10

The app uses the following libraries: 
- streamlit
- pandas
- numpy
- matplotlib

If you choose to clone the app and run it locally, you can install the requirements using the following command:
```
pip install -r requirements.txt
```
Note: If you create your own instance of the app on streamlit, the requirements will be automatically installed, but the requirements.txt file must be present.

## License
License details can be found in the [LICENSE](https://github.com/TempTrim/temp_trim/blob/main/LICENSE)

In general, 
- The app is free to use and modify for commercial or non-commercial purposes.
- The source **must be disclosed** if the app is modified and redistributed or re-used in any form.
- Please contact the authors if you'd like to contribute to the app and see the [Contributing](#contributing) section for more details.

## Contributing
The app is envisioned as a fully open source solution and we welcome suggestions and contributions to improve the soluton. The broaded vision is to enable public transit operators to apply similar models to their own systems and calculate savings in energy consumption for a variety of vehicles (trams, buses, trains, etc.).

Please follow the following best practices when contributing to the app:
- Submit your requests as [issues](https://github.com/TempTrim/temp_trim/issues) and allow the authors sufficient time to traiage the issue and respond.
- If you'd like to become a contributor, please contact us, we will be happy to add you. Follow standard procedures for pull requests as a contributor.
- Use the "backend" tag for issues related to the thermodynamic model and the "frontend" tag for issues related to the UI and UX of the app.

At the moment the largest priorities are: 
1. Extensive testing of the model 
2. Adding more types of vehicles
3. Improving the UI/X of the app.

## Developers
- [Florian Schubert](https://linkedin.com/in/f-schubert): Backend, Thermodynamic Model
- [Yash Dubey](https://www.linkedin.com/in/yashdubey132/): Frontend, UI and UX.



