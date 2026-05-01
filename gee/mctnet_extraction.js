/***************************************
 * mctnet_extraction.js
 *
 * Project Part 1 + Part 2:
 *   - Sentinel-2 L2A time series on 36 ten-day steps
 *   - CDL 2021 labels
 *   - ESA WorldCover 2021 cropland mask
 *   - Environmental covariates aligned to the same 36 steps
 *
 * Paper-aligned settings:
 *   - 36 observations at 10-day interval
 *   - 10 Sentinel-2 bands
 *   - CDL confidence >= 95
 *   - WorldCover cropland mask
 *   - 10,000 random points per state
 *   - classes < 5% merged into "others"
 *   - missing Sentinel-2 values marked with 0
 *
 * Important implementation note for Part 2:
 *   - Climate variables are genuinely temporal and aggregated on each 10-day window.
 *   - Soil and topography are static in the chosen GEE products. They are exported
 *     with suffixes _t01.._t36 so the downstream tensor shape is [36, 8].
 ***************************************/

var CONFIG = {
  year: 2021,
  startDate: '2021-01-01',
  endDate: '2022-01-01',
  stepDays: 10,
  nTimeSteps: 36,
  sampleCount: 10000,
  cdlConfidenceThreshold: 95,
  rareClassFraction: 0.05,
  exportFolder: 'mctnet_AR_CA_2021_fast',
  randomSeed: 2021,
  targetScale: 30,
  s2Bands: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
  climateBands: ['climate_pr_sum_mm', 'climate_tmmn_mean_c', 'climate_vpd_mean_kpa'],
  soilBands: ['soil_clay_0cm_pct', 'soil_sand_0cm_pct', 'soil_phh2o_0cm'],
  topographyBands: ['topo_elevation_m', 'topo_slope_deg'],
  states: [
    {
      name: 'Arkansas',
      abbrev: 'AR',
      point1: {lat: 35.9, lon: -91.7},
      point2: {lat: 34.8, lon: -92.3}
    },
    {
      name: 'California',
      abbrev: 'CA',
      point1: {lat: 38.8, lon: -121.7},
      point2: {lat: 36.2, lon: -119.6}
    }
  ]
};

var CDL_2021 = ee.ImageCollection('USDA/NASS/CDL')
  .filterDate(CONFIG.startDate, CONFIG.endDate)
  .first();
var WORLDCOVER_2021 = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map');
var TARGET_PROJECTION = CDL_2021.select('cropland').projection();


function propertyToList(value) {
  var valueType = ee.String(ee.Algorithms.ObjectType(value));
  return ee.List(ee.Algorithms.If(
    valueType.equals('List'),
    value,
    ee.String(value).split(',')
  ));
}


function normalizeCodeString(value) {
  return ee.String(value).trim();
}


var cdlClassKeys = propertyToList(CDL_2021.get('cropland_class_values')).map(function(value) {
  return normalizeCodeString(value);
});
var cdlClassNames = propertyToList(CDL_2021.get('cropland_class_names')).map(function(value) {
  return ee.String(value).trim();
});
var cdlClassDict = ee.Dictionary.fromLists(cdlClassKeys, cdlClassNames);


function buildRectangleFromCorners(stateConfig) {
  var lonMin = Math.min(stateConfig.point1.lon, stateConfig.point2.lon);
  var lonMax = Math.max(stateConfig.point1.lon, stateConfig.point2.lon);
  var latMin = Math.min(stateConfig.point1.lat, stateConfig.point2.lat);
  var latMax = Math.max(stateConfig.point1.lat, stateConfig.point2.lat);
  return ee.Geometry.Rectangle([lonMin, latMin, lonMax, latMax], null, false);
}


function stepSuffix(stepIndex) {
  return ee.Number(stepIndex).add(1).format('%02d');
}


function buildWindowStart(stepIndex) {
  return ee.Date(CONFIG.startDate).advance(ee.Number(stepIndex).multiply(CONFIG.stepDays), 'day');
}


function buildWindowEnd(stepIndex) {
  var windowStart = buildWindowStart(stepIndex);
  var nominalWindowEnd = windowStart.advance(CONFIG.stepDays, 'day');
  return ee.Date(ee.Algorithms.If(
    ee.Number(stepIndex).eq(CONFIG.nTimeSteps - 1),
    ee.Date(CONFIG.endDate),
    nominalWindowEnd
  ));
}


function featureNamesForStep(stepIndex, baseNames) {
  var suffix = stepSuffix(stepIndex);
  return ee.List(baseNames).map(function(baseName) {
    return ee.String(baseName).cat('_t').cat(suffix);
  });
}


function validBandNameForStep(stepIndex) {
  return ee.String('valid_t').cat(stepSuffix(stepIndex));
}


function reprojectContinuous(image) {
  return image
    .resample('bilinear')
    .reproject({crs: TARGET_PROJECTION, scale: CONFIG.targetScale})
    .toFloat();
}


function emptyMaskedS2Image() {
  return ee.Image.constant(ee.List.repeat(0, CONFIG.s2Bands.length))
    .rename(CONFIG.s2Bands)
    .toFloat()
    .updateMask(ee.Image.constant(0));
}


function maskSentinel2Clouds(image) {
  var qa60 = image.select('QA60');
  var scl = image.select('SCL');

  var qaMask = qa60.bitwiseAnd(1 << 10).eq(0)
    .and(qa60.bitwiseAnd(1 << 11).eq(0));

  var sclMask = scl.neq(0)
    .and(scl.neq(1))
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image.updateMask(qaMask).updateMask(sclMask);
}


function buildSentinel2Collection(region) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(CONFIG.startDate, CONFIG.endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskSentinel2Clouds)
    .select(CONFIG.s2Bands);
}


function buildSentinel2Step(stepIndex, s2Collection) {
  var windowStart = buildWindowStart(stepIndex);
  var windowEnd = buildWindowEnd(stepIndex);
  var windowCollection = s2Collection.filterDate(windowStart, windowEnd);

  var composite = ee.Image(ee.Algorithms.If(
    windowCollection.size().gt(0),
    windowCollection.median().select(CONFIG.s2Bands).toFloat(),
    emptyMaskedS2Image()
  ));

  var featureImage = composite.unmask(0).rename(featureNamesForStep(stepIndex, CONFIG.s2Bands));
  var validImage = composite.select(CONFIG.s2Bands[0]).mask()
    .rename([validBandNameForStep(stepIndex)])
    .unmask(0)
    .toByte();

  return featureImage.addBands(validImage);
}


function buildSentinel2Stack(region) {
  var s2Collection = buildSentinel2Collection(region);
  var stepIndices = ee.List.sequence(0, CONFIG.nTimeSteps - 1);
  var emptyImage = ee.Image.constant(0).rename('empty').select([]);

  return ee.Image(stepIndices.iterate(function(stepIndex, stacked) {
    return ee.Image(stacked).addBands(buildSentinel2Step(stepIndex, s2Collection));
  }, emptyImage));
}


function buildClimateStep(stepIndex, region) {
  var windowStart = buildWindowStart(stepIndex);
  var windowEnd = buildWindowEnd(stepIndex);
  var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterBounds(region)
    .filterDate(windowStart, windowEnd);

  var pr = ee.Image(ee.Algorithms.If(
    gridmet.size().gt(0),
    gridmet.select('pr').sum(),
    ee.Image.constant(0)
  )).rename('climate_pr_sum_mm');

  var tmmn = ee.Image(ee.Algorithms.If(
    gridmet.size().gt(0),
    gridmet.select('tmmn').mean().subtract(273.15),
    ee.Image.constant(0)
  )).rename('climate_tmmn_mean_c');

  var vpd = ee.Image(ee.Algorithms.If(
    gridmet.size().gt(0),
    gridmet.select('vpd').mean(),
    ee.Image.constant(0)
  )).rename('climate_vpd_mean_kpa');

  var climateImage = ee.Image.cat([pr, tmmn, vpd]).clip(region);
  return reprojectContinuous(climateImage).rename(featureNamesForStep(stepIndex, CONFIG.climateBands));
}


function buildStaticSoil(region) {
  var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02')
    .select('b0')
    .rename('soil_clay_0cm_pct');
  var sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02')
    .select('b0')
    .rename('soil_sand_0cm_pct');
  var ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
    .select('b0')
    .divide(10)
    .rename('soil_phh2o_0cm');

  return reprojectContinuous(ee.Image.cat([clay, sand, ph]).clip(region));
}


function buildStaticTopography(region) {
  var dem = ee.Image('USGS/SRTMGL1_003')
    .select('elevation')
    .clip(region)
    .rename('topo_elevation_m');
  var slope = ee.Terrain.slope(dem).rename('topo_slope_deg');
  return reprojectContinuous(ee.Image.cat([dem, slope]).clip(region));
}


function renameBandsForStep(image, stepIndex, bandNames) {
  return image.rename(featureNamesForStep(stepIndex, bandNames));
}


function buildEnvironmentalStack(region) {
  var staticSoil = buildStaticSoil(region);
  var staticTopography = buildStaticTopography(region);
  var stepIndices = ee.List.sequence(0, CONFIG.nTimeSteps - 1);
  var emptyImage = ee.Image.constant(0).rename('empty').select([]);

  return ee.Image(stepIndices.iterate(function(stepIndex, stacked) {
    var climateStep = buildClimateStep(stepIndex, region);
    var soilStep = renameBandsForStep(staticSoil, stepIndex, CONFIG.soilBands);
    var topoStep = renameBandsForStep(staticTopography, stepIndex, CONFIG.topographyBands);
    return ee.Image(stacked).addBands(climateStep).addBands(soilStep).addBands(topoStep);
  }, emptyImage));
}


function buildEligibleMask(region) {
  var cropMask = CDL_2021.select('cropland').gt(0);
  var confidenceMask = CDL_2021.select('confidence').gte(CONFIG.cdlConfidenceThreshold);
  var worldCoverCropland = WORLDCOVER_2021.eq(40);

  return cropMask.and(confidenceMask).and(worldCoverCropland).clip(region);
}


function getRetainedClassCodes(samples) {
  var histogram = ee.Dictionary(samples.aggregate_histogram('label_code'));
  var minCount = ee.Number(samples.size()).multiply(CONFIG.rareClassFraction);
  return histogram.keys().map(function(key) {
    key = ee.String(key);
    var count = ee.Number(histogram.get(key));
    return ee.Algorithms.If(count.gte(minCount), ee.Number.parse(key), null);
  }).removeAll([null]);
}


function addLabelMetadata(feature, retainedCodes) {
  var labelKey = normalizeCodeString(feature.get('label_code'));
  var labelCode = ee.Number.parse(labelKey);
  var labelName = ee.String(cdlClassDict.get(labelKey, 'unknown'));
  var keepLabel = ee.List(retainedCodes).contains(labelCode);
  var finalCode = ee.Number(ee.Algorithms.If(keepLabel, labelCode, 0));
  var finalName = ee.String(ee.Algorithms.If(keepLabel, labelName, 'others'));
  var coordinates = feature.geometry().coordinates();
  var sampleId = ee.String(feature.get('system:index'));

  return feature.set({
    sample_id: sampleId,
    longitude: coordinates.get(0),
    latitude: coordinates.get(1),
    label_name: labelName,
    label_final_code: finalCode,
    label_final_name: finalName
  });
}


function buildSamplePoints(stateConfig) {
  var region = buildRectangleFromCorners(stateConfig);
  var eligibleMask = buildEligibleMask(region);
  var samplingClass = ee.Image.constant(1).updateMask(eligibleMask).rename('sampling_class');
  var labelCode = CDL_2021.select('cropland').updateMask(eligibleMask).rename('label_code');
  var cdlConfidence = CDL_2021.select('confidence').updateMask(eligibleMask).rename('cdl_confidence');

  var rawPoints = ee.Image.cat([samplingClass, labelCode, cdlConfidence]).stratifiedSample({
    numPoints: CONFIG.sampleCount,
    classBand: 'sampling_class',
    region: region,
    scale: CONFIG.targetScale,
    seed: CONFIG.randomSeed,
    geometries: true,
    tileScale: 4
  }).map(function(feature) {
    return feature.set({
      state_name: stateConfig.name,
      state_abbrev: stateConfig.abbrev,
      roi_point1_lat: stateConfig.point1.lat,
      roi_point1_lon: stateConfig.point1.lon,
      roi_point2_lat: stateConfig.point2.lat,
      roi_point2_lon: stateConfig.point2.lon
    });
  });

  var retainedCodes = getRetainedClassCodes(rawPoints);
  var labeledPoints = rawPoints.map(function(feature) {
    return addLabelMetadata(feature, retainedCodes);
  });

  return {
    region: region,
    labeledPoints: labeledPoints,
    retainedCodes: retainedCodes
  };
}


function buildFeatureSelectors() {
  var selectors = [
    'sample_id',
    'state_name',
    'state_abbrev',
    'roi_point1_lat',
    'roi_point1_lon',
    'roi_point2_lat',
    'roi_point2_lon',
    'longitude',
    'latitude'
  ];

  for (var step = 1; step <= CONFIG.nTimeSteps; step++) {
    var suffix = ('0' + step).slice(-2);
    CONFIG.s2Bands.forEach(function(bandName) {
      selectors.push(bandName + '_t' + suffix);
    });
    selectors.push('valid_t' + suffix);
  }

  CONFIG.climateBands.forEach(function(bandName) {
    for (var stepIdx = 1; stepIdx <= CONFIG.nTimeSteps; stepIdx++) {
      selectors.push(bandName + '_t' + ('0' + stepIdx).slice(-2));
    }
  });
  CONFIG.soilBands.forEach(function(bandName) {
    for (var stepIdx2 = 1; stepIdx2 <= CONFIG.nTimeSteps; stepIdx2++) {
      selectors.push(bandName + '_t' + ('0' + stepIdx2).slice(-2));
    }
  });
  CONFIG.topographyBands.forEach(function(bandName) {
    for (var stepIdx3 = 1; stepIdx3 <= CONFIG.nTimeSteps; stepIdx3++) {
      selectors.push(bandName + '_t' + ('0' + stepIdx3).slice(-2));
    }
  });

  return selectors;
}


function buildLabelSelectors() {
  return [
    'sample_id',
    'state_name',
    'state_abbrev',
    'roi_point1_lat',
    'roi_point1_lon',
    'roi_point2_lat',
    'roi_point2_lon',
    'longitude',
    'latitude',
    'cdl_confidence',
    'label_code',
    'label_name',
    'label_final_code',
    'label_final_name'
  ];
}


function exportState(stateConfig) {
  var sampleData = buildSamplePoints(stateConfig);
  var region = sampleData.region;
  var featureImage = buildSentinel2Stack(region).addBands(buildEnvironmentalStack(region));

  var featureSamples = featureImage.sampleRegions({
    collection: sampleData.labeledPoints,
    properties: [
      'sample_id',
      'state_name',
      'state_abbrev',
      'roi_point1_lat',
      'roi_point1_lon',
      'roi_point2_lat',
      'roi_point2_lon',
      'longitude',
      'latitude'
    ],
    scale: CONFIG.targetScale,
    geometries: false,
    tileScale: 4
  });

  var labelSamples = sampleData.labeledPoints.map(function(feature) {
    return ee.Feature(null, feature.toDictionary(buildLabelSelectors()));
  });

  print('State:', stateConfig.name);
  print('Retained class codes:', sampleData.retainedCodes);
  print('Final class histogram (' + stateConfig.abbrev + '):', labelSamples.aggregate_histogram('label_final_name'));

  Map.addLayer(region, {}, stateConfig.name + ' ROI', false);
  Map.addLayer(
    WORLDCOVER_2021.eq(40).clip(region),
    {min: 0, max: 1, palette: ['000000', '00ff00']},
    stateConfig.name + ' WorldCover cropland',
    false
  );

  Export.table.toDrive({
    collection: featureSamples.select(buildFeatureSelectors()),
    description: 'mctnet_' + stateConfig.abbrev + '_' + CONFIG.year,
    folder: CONFIG.exportFolder,
    fileNamePrefix: 'mctnet_' + stateConfig.abbrev + '_' + CONFIG.year,
    fileFormat: 'CSV'
  });

  Export.table.toDrive({
    collection: labelSamples.select(buildLabelSelectors()),
    description: 'mctnet_samples_' + stateConfig.abbrev + '_' + CONFIG.year,
    folder: CONFIG.exportFolder,
    fileNamePrefix: 'mctnet_samples_' + stateConfig.abbrev + '_' + CONFIG.year,
    fileFormat: 'CSV'
  });
}


CONFIG.states.forEach(function(stateConfig) {
  exportState(stateConfig);
});

Map.setCenter(-98.5, 36.0, 4);
