import { useMemo } from 'react';
import { TelemetrySample } from '../App';

type Props = {
  samples: TelemetrySample[];
  latest?: TelemetrySample;
};

function GpsPanel({ samples, latest }: Props) {
  const bounds = useMemo(() => {
    const lats = samples.map((s) => s.gps.lat);
    const lngs = samples.map((s) => s.gps.lng);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs);
    const maxLng = Math.max(...lngs);
    return { minLat, maxLat, minLng, maxLng };
  }, [samples]);

  const normalizePoint = (sample: TelemetrySample) => {
    const latRange = bounds.maxLat - bounds.minLat || 1;
    const lngRange = bounds.maxLng - bounds.minLng || 1;
    return {
      x: ((sample.gps.lng - bounds.minLng) / lngRange) * 100,
      y: 100 - ((sample.gps.lat - bounds.minLat) / latRange) * 100
    };
  };

  return (
    <article className="panel map">
      <div className="panel-header">
        <h2>GPS trace</h2>
        <p className="muted">Scaled view of the latest filtered trip.</p>
      </div>
      <div className="map-frame">
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id="pathGradient" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#22d3ee" />
            </linearGradient>
          </defs>
          <rect width="100" height="100" rx="8" fill="#0f172a" />
          {samples.map((sample, index) => {
            const point = normalizePoint(sample);
            return (
              <circle
                key={sample.timestamp}
                cx={point.x}
                cy={point.y}
                r={index === samples.length - 1 ? 2.6 : 1.8}
                fill={index === samples.length - 1 ? '#f59e0b' : 'url(#pathGradient)'}
                opacity={0.9}
              />
            );
          })}
        </svg>
      </div>
      {latest && (
        <div className="map-footer">
          <div>
            <p className="muted">Now tracking</p>
            <p className="strong">{latest.gps.lat.toFixed(5)}, {latest.gps.lng.toFixed(5)}</p>
          </div>
          <div>
            <p className="muted">Speed</p>
            <p className="strong">{latest.gps.speedKph.toFixed(1)} km/h</p>
          </div>
        </div>
      )}
    </article>
  );
}

export default GpsPanel;
