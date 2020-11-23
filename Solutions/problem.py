import pandas as pd
import numpy as np
import sys
from scipy.optimize import curve_fit

'''
This script takes the following information about a medical practice as input:
    1. CSV filename of daily visiting volume (cols 'Date', 'Counts')
    2. CSV filename of daily COVID-19 screening volume (cols 'Date', 'Counts')
    3. Integer value of COVID-19 screening threshold
From these inputs, the script fits a predictive curve of the maximum projected 
number of COVID-19 screens over the next 31 days. If any days exceed the 
threshold provided, return "True".

Call the function from the command line using the following command:

python3 problem.py ${visitCsv} ${screenCsv} ${screenThresh}

Features still to implement:
    * Daily breakdown of likelihoods of exceeding the threshold provided
    * Argparser logic for cleaner command line usage

Requirements:
numpy==1.19.3
scipy==1.5.4
pandas==1.1.4
'''

def readVisitCSV(fn):
    '''
    Pandas read_csv wrapper that initializes a 'Date' column as datetime.
    '''
    df = pd.read_csv(fn)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def sqrtFitFn(x, a, b, c):
    return a*np.sqrt(x-b)+c

def polyOrd6Fn(x, a, b, c, d, e, f, g):
    return np.polyval([a, b, c, d, e, f, g], x)

def createScreenRateDf(screenDf, visitDf):
    '''
    Create a dataframe of the daily rates of COVID-19 screening as a fraction
    of total visits to correct for weekend drops in visits.
    '''
    screenToVisitDf = screenDf.copy()
    screenToVisitDf['Counts'] = screenDf['Counts']/visitDf['Counts']

    # Remove data for dates where no COVID-19 screening occurred.
    cleanedScreenRatioDf = screenToVisitDf[screenToVisitDf.Counts != 0]
    return cleanedScreenRatioDf

def fitScreenRate(screenRateDf):
    '''
    Fit & extrapolate the rate of COVID-19 screens for the range given.
    '''
    x = screenRateDf.index
    y = screenRateDf['Counts']

    # Initialize bounds for the first day of COVID-19 screening
    loDay = np.floor(x[0]/10)*10
    hiDay = np.ceil(x[0]/10)*10

    # Fit to a square root curve & return optimized parameters.
    scrRateBounds = ([-np.inf, loDay, -np.inf], [np.inf, hiDay, np.inf])
    scrRatePopt, scrRatePcov = curve_fit(sqrtFitFn, x, y, bounds=scrRateBounds)

    return (scrRatePopt, scrRatePcov)


def fitVisits(visitDf):
    '''
    Fit & extrapolate the number of visits per day for the range given.
    '''

    # Exclude weekends from fit due to substantially lower visits
    weekdayVisitDf = visitDf.loc[visitDf['Date'].dt.dayofweek.lt(5)]
    x = weekdayVisitDf.index
    y = weekdayVisitDf['Counts']

    # Fit to a polynomial curve
    visitPopts, visitPcov = curve_fit(polyOrd6Fn, x, y)
    return visitPopts, visitPcov


def plotProjected(projectedDateDf, projectedScreenCurve):
    '''
    Plot the an extrapolated curve (red) over the date range provided.
    '''
    projectedDateDf.plot(
        x='Date',
        y='Counts',
        ylabel='Visit Count',
        title="Projected COVID-19 Screenings"
    ).plot(
        projectedDateDf['Date'],
        projectedScreenCurve,
        color='red'
    )

def extendDatesDf(df, timeDelta):
    '''
    Extend the date range of the original df by reindexing on the Date index, 
    then resetting the Date index from the start.
    '''
    startDate = df['Date'].iat[0].date()
    endDate = (df['Date'].iat[-1] + pd.to_timedelta(timeDelta)).date()
    extendedDates = pd.date_range(start=startDate, end=endDate, freq='D')
    projectedDateDf = df.set_index('Date')
    projectedDateDf = projectedDateDf.reindex(extendedDates)
    projectedDateDf = projectedDateDf.reset_index().rename(
            columns={'index':'Date'})

    return projectedDateDf

def main(visitCsv, screenCsv, screenThresh):
    screenDf = readVisitCSV(screenCsv)
    visitDf = readVisitCSV(visitCsv)
    projectedDays = 31
    projectedDateDf = extendDatesDf(screenDf, f'{projectedDays}D')

    # Establish number of standard deviations for calculating fit errors
    nstd = 1.0

    # Prepare curve fit for screening visits / total visits over time
    screenRateDf = createScreenRateDf(screenDf, visitDf)
    scrRatePopt, scrRatePcov = fitScreenRate(screenRateDf)
    scrRatePerr = np.sqrt(np.diag(scrRatePcov))
    scrRateCurve = sqrtFitFn(projectedDateDf.index, *scrRatePopt)

    maxScrRatePopt = scrRatePopt + nstd*scrRatePerr
    maxScrRateCurve = sqrtFitFn(projectedDateDf.index, *maxScrRatePopt)

    # Prepare curve fit for total visits over time
    cleanedVisitDf = visitDf.loc[screenRateDf.index]
    visitPopt, visitPcov = fitVisits(cleanedVisitDf)
    visitPerr = np.sqrt(np.diag(visitPcov))
    visitCurve = np.polyval(visitPopt, projectedDateDf.index)

    maxVisitPopt = visitPopt + nstd*visitPerr
    maxVisitCurve = np.polyval(maxVisitPopt, projectedDateDf.index)

    # Project gestalt curve of expected number of visits * expected rate of 
    # COVID-19 screens.
    projectedScreenCurve = scrRateCurve*visitCurve
    maxProjectedCurve = maxScrRateCurve*maxVisitCurve
    plotProjected(projectedDateDf, projectedScreenCurve)

    # Check if any projected days exceed the screening threshold
    overScreenThresh = pd.Series(maxProjectedCurve).gt(screenThresh).any()

    # Not enough time to accurately calculate likelihoods, omitting for now.
    return overScreenThresh, np.nan

if __name__ == "__main__":
    visitCsv = sys.argv[1]
    screenCsv = sys.argv[2]
    screenThresh = sys.argv[3]
    print(main(visitCsv, screenCsv, screenThresh))