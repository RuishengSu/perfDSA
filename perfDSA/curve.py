import traces


def resample(time_axis, value_axis, sampling_period=250):
    ts = traces.TimeSeries(zip(time_axis, value_axis))
    resampled_tuples = ts.sample(sampling_period=sampling_period, interpolate='linear')
    resampled_values = [t[1] for t in resampled_tuples]
    return resampled_values
