import { TelemetrySample } from '../App';

type Props = {
  samples: TelemetrySample[];
};

function DataTable({ samples }: Props) {
  return (
    <article className="panel">
      <div className="panel-header">
        <h2>Raw samples</h2>
        <p className="muted">Tap any row to compare against the reference ride.</p>
      </div>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Speed (km/h)</th>
              <th>Accel (ax, ay, az)</th>
              <th>Battery (%)</th>
              <th>Event</th>
            </tr>
          </thead>
          <tbody>
            {samples.map((sample) => (
              <tr key={sample.timestamp}>
                <td>{new Date(sample.timestamp).toLocaleTimeString()}</td>
                <td>{sample.gps.speedKph.toFixed(1)}</td>
                <td>
                  {sample.imu.ax.toFixed(2)}, {sample.imu.ay.toFixed(2)},{' '}
                  {sample.imu.az.toFixed(2)}
                </td>
                <td>{sample.battery}%</td>
                <td>{sample.event}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export default DataTable;
