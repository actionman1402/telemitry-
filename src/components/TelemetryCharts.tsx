import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { TelemetrySample } from '../App';

function formatTime(timestamp: string) {
  const date = new Date(timestamp);
  return `${date.getMinutes()}:${date.getSeconds().toString().padStart(2, '0')}`;
}

type Props = {
  samples: TelemetrySample[];
};

function TelemetryCharts({ samples }: Props) {
  const data = samples.map((sample) => ({
    time: formatTime(sample.timestamp),
    speed: sample.gps.speedKph,
    ax: sample.imu.ax,
    ay: sample.imu.ay,
    az: sample.imu.az,
    battery: sample.battery
  }));

  return (
    <div className="chart-stack">
      <article className="panel">
        <div className="panel-header">
          <h2>Speed + battery</h2>
          <p className="muted">Trendlines for quick QA against the BYB baseline.</p>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data} margin={{ top: 20, left: 0, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis yAxisId="speed" stroke="#7c3aed" />
            <YAxis yAxisId="battery" orientation="right" stroke="#22d3ee" />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="speed"
              type="monotone"
              dataKey="speed"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
            <Line
              yAxisId="battery"
              type="monotone"
              dataKey="battery"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </article>

      <article className="panel">
        <div className="panel-header">
          <h2>IMU triad</h2>
          <p className="muted">Acceleration in device frame; compare jitter against tolerances.</p>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={data} margin={{ top: 20, left: 0, right: 10 }}>
            <defs>
              <linearGradient id="imuX" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="imuY" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="imuZ" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="ax" stroke="#ef4444" fill="url(#imuX)" />
            <Area type="monotone" dataKey="ay" stroke="#22c55e" fill="url(#imuY)" />
            <Area type="monotone" dataKey="az" stroke="#06b6d4" fill="url(#imuZ)" />
          </AreaChart>
        </ResponsiveContainer>
      </article>
    </div>
  );
}

export default TelemetryCharts;
