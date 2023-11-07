import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("C:\D\Telo🌙\python\modulo3\STREAMLIT\ml_airbnb_sevilla")
st.title("Sistema de predicción de precios Upgrade-Hub")

neighbourhood = st.selectbox('Barrio', options=['San Vicente' ,'San Lorenzo' ,'Feria' ,'Arenal' ,'Santa Cruz',
 'León XIII', 'Los Naranjos', 'Las Avenidas', 'Museo', 'Alfalfa', 'San Bernardo',
 'El Cerro' ,'Encarnación', 'Regina' ,'an Roque', 'Triana Casco Antiguo',
 'Macarena 3 Huertas', 'Macarena 5' ,'Triana Este', 'San Bartolomé',
 'Triana Oeste', 'San Gil', 'El Porvenir', 'Santa Catalina' ,'San Julián',
 'Pino Flores', 'Sector Sur', 'La Palmera', 'Reina Mercedes',
 'El Torrejón', 'El Cerezo', 'Doctor Barraquer','G. Renfe, Policlínico',
 'Santa María de Ordas, San Nicolas', 'Huerta de la Salud', 'Santa Clara',
 'San José Obrero', 'La Florida' ,'Los Remedios',
 'Palacio Congresos',' Urbadiez', 'Entrepuentes',' Jardines del Eden',
 'Begoña, Santa Catalina' ,'Valdezorras', 'La Calzada' ,'San Pablo A y B'
 'La Bachillera', 'Pedro Salvador',' Las Palmeritas' ,'El Tardón', 'El Carmen',
 'El Plantinar' ,'Cruz Roja, Capuchinos' ,'Los Pajaros',
 'Cisneo Alto, Santa María de Graccia', 'Arbol Gordo' ,'Bellavista',
 'Consolación', 'Ciudad Jardín' ,'Nervión', 'Heliópolis', 'Polígono Sur',
 'no asignado' ,'La Buhaira' ,'Parque Alcosa', 'Huerta del Pilar' ,'Tablada',
 'Carretera de Carmona, María Auxiliadora, Fontanal', 'San Jerónimo',
 'Tiro de Línea, Santa Genoveva', 'La Paz, Las Golondrinas' ,'Retiro Obrero',
 'Bda. Pino Montano', 'Barrio León' ,'Colores, Entreparques',
 'Huerta de Santa Teresa', 'Bami', 'Giralda Sur' ,'Las Huertas',
 'El Cano, Los Bermejales', 'Pio XII' ,'San Diego' ,'San Pablo C',
 'Hermandades, La Carrasca' ,'Los Carteros' ,'San Pablo D y E' ,'La Corza',
 'Polígono Norte', 'El Carmen' ,'Campos de Soria' ,'La Barzola',
 'Santa Aurelia, Cantábrico, Atlántico, La Romería',
 'La Palmilla, Doctor Marañón', 'Palmete', 'Aeropuerto Viejo',
 'Felipe II, Los Diez Mandamientos', 'Los Príncipes, La Fontanilla',
 'El Juncal, Híspalis' ,'Villegas', 'Avda. de la Paz' 'El Gordillo',
 'Juan XXIII', 'Amate' ,'Prado, Parque María Luisa', 'Bda. de Pineda',
 'Zodiaco', 'San Carlos, Tartessos', 'La Plata', 'Rochelambert' ,'La Oliva',
 'Las Letanías' ,'Los arcos', 'El Rocío' ,'Tabladilla, La Estrella',
 'Las Naciones, Parque Atlántico, Las Dalias', 'San Matías'])

property_type = st.selectbox('Tipo de Propiedad', options=[
'Entire rental unit' ,'Entire loft' ,'Entire condo', 'Entire home',
 'Private room in rental unit', 'Private room in tiny home',
 'Private room in townhouse', 'Entire serviced apartment',
 'Room in serviced apartment' ,'Entire chalet' ,'Entire villa',
 'Entire townhouse', 'Entire vacation home', 'Private room in home',
 'Private room in bed and breakfast' ,'Private room in serviced apartment',
 'Room in hotel', 'Private room in hostel' ,'Private room in condo',
 'Entire cottage' ,'Entire guesthouse' ,'Entire guest suite',
 'Private room in villa', 'Private room in loft', 'Room in aparthotel',
 'Private room' ,'Shared room in rental unit' ,'Private room in guesthouse',
 'Private room in casa particular', 'Entire place',
 'Private room in earthen home', 'Casa particular' ,'Shared room in home',
 'Private room in guest suite', 'Room in bed and breakfast',
 'Room in boutique hotel', 'Tiny home' ,'Shared room in hostel',
 'Private room in dome', 'Shared room in bed and breakfast',
 'Shared room in serviced apartment', 'Castle',
 'Shared room in vacation home', 'Shared room in hotel' ,'Camper/RV'
])

accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'])
host_is_superhost = st.selectbox('Host es SuperHost', options=['t','f'])
minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    host_is_superhost, minimum_nights
]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'host_is_superhost', 'minimum_nights'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')