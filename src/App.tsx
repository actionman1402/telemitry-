import { useMemo, useState } from 'react';
import telemetryRaw from '../data/mockTelemetry.json';
import DataTable from './components/DataTable';
import GpsPanel from './components/GpsPanel';
import TelemetryCharts from './components/TelemetryCharts';
import Toolbar from './components/Toolbar';
import MetricCard from './components/MetricCard';

export type TelemetrySample = {
  timestamp: string;
  gps: { lat: number; lng: number; speedKph: number };
  imu: { ax: number; ay: number; az: number };
  battery: number;
  event: string;
};

const telemetry = telemetryRaw as TelemetrySample[];

function App() {
  const [samples, setSamples] = useState<TelemetrySample[]>(telemetry);
  const [eventFilter, setEventFilter] = useState<string>('all');
  const [loadError, setLoadError] = useState<string | null>(null);

  const filteredData = useMemo(() => {
    const source = eventFilter === 'all'
      ? samples
      : samples.filter((sample) => sample.event === eventFilter);

    return source.length ? source : samples;
  }, [eventFilter, samples]);

  const avgSpeed = useMemo(
    () =>
      filteredData.reduce((acc, sample) => acc + sample.gps.speedKph, 0) /
      Math.max(filteredData.length, 1),
    [filteredData]
  );

  const avgBattery = useMemo(
    () =>
      filteredData.reduce((acc, sample) => acc + sample.battery, 0) /
      Math.max(filteredData.length, 1),
    [filteredData]
  );

  const latest = filteredData[filteredData.length - 1] ?? samples[0];

  const handleImport = async (file: File) => {
    try {
      const content = await file.text();
      const parsed = JSON.parse(content);

      if (!Array.isArray(parsed)) {
        throw new Error('File must be an array of telemetry samples');
      }

      setSamples(parsed as TelemetrySample[]);
      setEventFilter('all');
      setLoadError(null);
    } catch (error) {
      console.error('Failed to import telemetry', error);
      setSamples(telemetry);
      setEventFilter('all');
      setLoadError('Unable to import file. Expected JSON array exported from Motion IQ telemetry. Reverted to bundled mock data.');
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Motion IQ</p>
          <h1>Telemitry Viewer</h1>
          <p className="lede">
            Inspect Motion IQ telemity sessions, visualize GPS tracks, and review IMU
            and battery trends. The project ships with mock data but you can wire it
            to live feeds from your motion sensor gateway.
          </p>
        </div>
        <Toolbar value={eventFilter} onChange={setEventFilter} onImport={handleImport} />
      </header>

      {loadError && <p className="error-banner">{loadError}</p>}

      <section className="metrics">
        <MetricCard
          label="Average speed"
          value={`${avgSpeed.toFixed(1)} km/h`}
          hint="Based on filtered window"
        />
        <MetricCard
          label="Average battery"
          value={`${avgBattery.toFixed(0)} %`}
          hint="Device health"
        />
        <MetricCard
          label="Latest event"
          value={latest?.event ?? 'n/a'}
          hint={new Date(latest?.timestamp ?? '').toLocaleTimeString()}
        />
      </section>

      <section className="grid">
        <GpsPanel latest={latest} samples={filteredData} />
        <TelemetryCharts samples={filteredData} />
        <DataTable samples={filteredData} />
      </section>
    </div>
  );
}

export default App;
