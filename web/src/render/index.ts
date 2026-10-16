/** Shared rendering toolkit for the scale views. */
export { createScene, type SceneHandle, type SceneOptions } from './scene';
export { viridis, coolwarm, sampleColormap, colormapGradient, LUT_SIZE, type Colormap } from './colormap';
export { createColormapTexture, setRange, setColormap, disposeObject, disposeMaterial } from './gpu';
export { autoRange, type Range } from './range';
export { extractSlice, sliceDims, fieldIndex, type Axis, type Shape3 } from './field';
export { SlicePlane, type SliceOptions } from './slice';
export { SurfaceField } from './surface';
export { PointCloud, type PointCloudOptions } from './points';
export { Colorbar, type ColorbarOptions } from './colorbar';
export { RingBuffer } from './ringbuffer';
export { TimeSeriesPlot, type TimeSeriesOptions, type TraceOptions } from './timeseries';
export { niceTicks, formatTick } from './ticks';
