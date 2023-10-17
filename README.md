# Curve fitting the Two-Carrier Transport Induced Hall Anomaly in Graphene data 

As per [literature](https://pubs.acs.org/doi/full/10.1021/acsnano.6b01568) The evolution of the Hall curves with temperature shows a transition from p-
type to n-type conduction with increasing temperature. The observed Hall anomaly is a characteristic of two-carrier
transport, which can be explained by the two-carrier model described below :

![image](https://github.com/DRA-chaos/Curve-fitting-for-graphene-devices-based-on-the-two-carrier-model/assets/68393451/945a6e83-ec44-462e-b2c6-147f5614982b)

where ne(nh) and μe(μh) are the carrier density and mobility of electrons (holes), respectively. According to the first equation, the Hall resistivity reverses its sign 
at a critical magnetic field Bc. In case of μe > μh, the increase of electron concentration with increasing temperature will lead to an increase int the critical magnetic field.

The [script](https://github.com/DRA-chaos/Curve-fitting-for-graphene-devices-based-on-the-two-carrier-model/blob/main/Curve%20Fitting%20scripts/two%20band%20model%20graphene.py) is written using the Differential evolution algorithm for global optimization. Machine Learning , Deep learning and other advanced data science frameworks could 
not be deployed because of the scanty number of data points ( merely 50) obtained from the Hall probe making this automated curve fitting script a tricky problem owing to weighted outliers creeping up from experimental conditions like temperature and pressure of the cryostat. The data can be made available upon request, 
please contact Dr Karthik V Raman at TIFR-H for the same.
