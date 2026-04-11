/***************************************
 * MCTNet base data preparation in GEE
 * Paper: Wang et al. (2024)
 * "A lightweight CNN-Transformer network for pixel-based crop mapping
 * using time-series Sentinel-2 imagery"
 *
 * This script prepares two user-defined study areas:
 *   - Arkansas ROI
 *   - California ROI
 *
 * Exports:
 *   - One CSV per state with 10,000 randomly sampled labeled points
 *   - 360 Sentinel-2 spectral features (36 time steps x 10 bands)
 *   - 36 validity flags to mark missing temporal observations
 *
 * Requirements:
 *   - Google Earth Engine account and activated access
 *   - Run inside the Earth Engine Code Editor
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
  exportFolder: 'mctnet_crop_mapping_2021',
  randomSeed: 2021,
  // GEE expects signed decimal degrees. Western longitudes are negative.
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
  ],
  s2Bands: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
};

var CDL_2021 = ee.ImageCollection('USDA/NASS/CDL')
  .filterDate(CONFIG.startDate, CONFIG.endDate)
  .first();
var WORLDCOVER_2021 = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map');

function propertyToEeList(propertyValue) {
  var propertyType = ee.String(ee.Algorithms.ObjectType(propertyValue));
  return ee.List(ee.Algorithms.If(
    ee.Algorithms.IsEqual(propertyType, 'List'),
    propertyValue,
    ee.String(propertyValue).split(',')
  ));
}

function normalizeCodeString(value) {
  return ee.String(value).trim();
}

var cdlClassKeys = propertyToEeList(CDL_2021.get('cropland_class_values')).map(function(value) {
  return normalizeCodeString(value);
});
var cdlClassNames = propertyToEeList(CDL_2021.get('cropland_class_names')).map(function(name) {
  return ee.String(name);
});
var cdlClassDict = ee.Dictionary.fromLists(cdlClassKeys, cdlClassNames);

function buildRegionFromCornerPoints(stateConfig) {
  var lonMin = Math.min(stateConfig.point1.lon, stateConfig.point2.lon);
  var lonMax = Math.max(stateConfig.point1.lon, stateConfig.point2.lon);
  var latMin = Math.min(stateConfig.point1.lat, stateConfig.point2.lat);
  var latMax = Math.max(stateConfig.point1.lat, stateConfig.point2.lat);

  return ee.Geometry.Rectangle([lonMin, latMin, lonMax, latMax], null, false);
}

function formatStep(stepIndex) {
  return ee.Number(stepIndex).add(1).format('%02d');
}

function getFeatureBandNamesForStep(stepIndex) {
  var stepString = formatStep(stepIndex);
  return ee.List(CONFIG.s2Bands).map(function(bandName) {
    return ee.String(bandName).cat('_t').cat(stepString);
  });
}

function getValidBandNameForStep(stepIndex) {
  return ee.String('valid_t').cat(formatStep(stepIndex));
}

function getExportSelectors() {
  var selectors = [
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

  for (var step = 1; step <= CONFIG.nTimeSteps; step++) {
    var stepString = ('0' + step).slice(-2);
    CONFIG.s2Bands.forEach(function(bandName) {
      selectors.push(bandName + '_t' + stepString);
    });
    selectors.push('valid_t' + stepString);
  }

  return selectors;
}

function makeFullyMaskedEmptyImage() {
  return ee.Image.constant(ee.List.repeat(0, CONFIG.s2Bands.length))
    .rename(CONFIG.s2Bands)
    .toFloat()
    .updateMask(ee.Image.constant(0));
}

function maskSentinel2Clouds(image) {
  // Implementation choice:
  // The paper only says "cloud-affected pixels were eliminated".
  // Here we use QA60 + SCL from Sentinel-2 L2A to remove clouds,
  // cirrus, shadows, snow/ice, saturated pixels, and no-data pixels.
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

function buildTenDayComposite(stepIndex, s2Collection) {
  stepIndex = ee.Number(stepIndex);
  var windowStart = ee.Date(CONFIG.startDate).advance(
    stepIndex.multiply(CONFIG.stepDays),
    'day'
  );
  var nominalWindowEnd = windowStart.advance(CONFIG.stepDays, 'day');
  var windowEnd = ee.Date(ee.Algorithms.If(
    stepIndex.eq(CONFIG.nTimeSteps - 1),
    ee.Date(CONFIG.endDate),
    nominalWindowEnd
  ));

  var windowCollection = s2Collection.filterDate(windowStart, windowEnd);
  var composite = ee.Image(
    ee.Algorithms.If(
      windowCollection.size().gt(0),
      windowCollection.median().select(CONFIG.s2Bands).toFloat(),
      makeFullyMaskedEmptyImage()
    )
  );

  // Paper: missing values are kept and marked as 0.
  var featureImage = composite.unmask(0).rename(getFeatureBandNamesForStep(stepIndex));

  // Paper: Input 2 indicates whether the temporal observation is missing.
  // We export one validity flag per time step (1 = observed, 0 = missing).
  var validImage = composite.select(CONFIG.s2Bands[0]).mask()
    .rename([getValidBandNameForStep(stepIndex)])
    .unmask(0)
    .toByte();

  return featureImage.addBands(validImage);
}

function buildTimeSeriesStack(region) {
  var s2Collection = buildSentinel2Collection(region);
  var stepIndices = ee.List.sequence(0, CONFIG.nTimeSteps - 1);
  var emptyImage = ee.Image.constant(0).rename('empty').select([]);

  return ee.Image(stepIndices.iterate(function(stepIndex, stacked) {
    var composite = buildTenDayComposite(stepIndex, s2Collection);
    return ee.Image(stacked).addBands(composite);
  }, emptyImage));
}

function buildEligibleMask(region) {
  var cdlCropland = CDL_2021.select('cropland');
  var cdlConfidence = CDL_2021.select('confidence');
  var worldCoverCropland = WORLDCOVER_2021.eq(40);

  return cdlCropland.gt(0)
    .and(cdlConfidence.gte(CONFIG.cdlConfidenceThreshold))
    .and(worldCoverCropland)
    .clip(region);
}

function getRetainedClassCodes(sampleCollection) {
  var histogram = ee.Dictionary(sampleCollection.aggregate_histogram('label_code'));
  var minCount = ee.Number(CONFIG.sampleCount).multiply(CONFIG.rareClassFraction);

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

  return feature.set({
    sample_id: ee.String(feature.get('system:index')),
    longitude: coordinates.get(0),
    latitude: coordinates.get(1),
    label_name: labelName,
    label_final_code: finalCode,
    label_final_name: finalName
  });
}

function buildStateSamples(stateConfig) {
  var region = buildRegionFromCornerPoints(stateConfig);
  var regionFeature = ee.Feature(region, {
    state_name: stateConfig.name,
    state_abbrev: stateConfig.abbrev
  });
  var eligibleMask = buildEligibleMask(region);
  var timeSeriesStack = buildTimeSeriesStack(region);
  var cdlCropland = CDL_2021.select('cropland');
  var cdlConfidence = CDL_2021.select('confidence');
  var samplingClass = ee.Image.constant(1).updateMask(eligibleMask).rename('sampling_class');

  var samplingImage = timeSeriesStack
    .addBands(cdlCropland.updateMask(eligibleMask).rename('label_code'))
    .addBands(cdlConfidence.updateMask(eligibleMask).rename('cdl_confidence'))
    .addBands(samplingClass);

  // Paper: 10,000 random points per study area.
  var rawSamples = samplingImage.stratifiedSample({
    numPoints: CONFIG.sampleCount,
    classBand: 'sampling_class',
    region: region,
    scale: 30,
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

  var retainedCodes = getRetainedClassCodes(rawSamples);
  var finalSamples = rawSamples
    .map(function(feature) {
      return addLabelMetadata(feature, retainedCodes);
    })
    .map(function(feature) {
      return feature.setGeometry(null);
    });

  return {
    regionFeature: regionFeature,
    stack: timeSeriesStack,
    samples: finalSamples,
    retainedCodes: retainedCodes
  };
}

function exportStateSamples(stateConfig) {
  var stateData = buildStateSamples(stateConfig);
  var selectors = getExportSelectors();

  print('Study area:', stateConfig.name);
  print('ROI point 1:', stateConfig.point1);
  print('ROI point 2:', stateConfig.point2);
  print(
    'Raw CDL histogram (' + stateConfig.abbrev + '):',
    stateData.samples.aggregate_histogram('label_code')
  );
  print(
    'Final class histogram (' + stateConfig.abbrev + '):',
    stateData.samples.aggregate_histogram('label_final_name')
  );
  print(
    'Retained class codes (' + stateConfig.abbrev + '):',
    stateData.retainedCodes
  );

  Map.addLayer(stateData.regionFeature, {}, stateConfig.name + ' ROI', false);
  Map.addLayer(
    WORLDCOVER_2021.eq(40).clip(stateData.regionFeature.geometry()),
    {min: 0, max: 1, palette: ['000000', '00FF00']},
    stateConfig.name + ' WorldCover cropland',
    false
  );
  Map.addLayer(
    CDL_2021.select('cropland').clip(stateData.regionFeature.geometry()),
    {},
    stateConfig.name + ' CDL 2021',
    false
  );

  Export.table.toDrive({
    collection: stateData.samples.select(selectors),
    description: 'mctnet_samples_' + stateConfig.abbrev + '_' + CONFIG.year,
    folder: CONFIG.exportFolder,
    fileNamePrefix: 'mctnet_samples_' + stateConfig.abbrev + '_' + CONFIG.year,
    fileFormat: 'CSV'
  });
}

CONFIG.states.forEach(function(stateConfig) {
  exportStateSamples(stateConfig);
});

Map.setCenter(-98.5, 36.0, 4);
