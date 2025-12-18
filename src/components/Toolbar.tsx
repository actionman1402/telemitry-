import { ChangeEvent } from 'react';

type Props = {
  value: string;
  onChange: (value: string) => void;
  onImport: (file: File) => void;
};

const filters = [
  { label: 'All events', value: 'all' },
  { label: 'Start', value: 'start' },
  { label: 'Record', value: 'record' }
];

function Toolbar({ value, onChange, onImport }: Props) {
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const [file] = Array.from(event.target.files ?? []);
    if (file) {
      onImport(file);
      event.target.value = '';
    }
  };

  const triggerFilePicker = () => {
    const input = document.getElementById('telemitry-import-input') as HTMLInputElement | null;
    input?.click();
  };

  return (
    <div className="toolbar">
      <span className="muted">Event filter</span>
      <div className="pill-group">
        {filters.map((filter) => (
          <button
            key={filter.value}
            className={value === filter.value ? 'pill active' : 'pill'}
            onClick={() => onChange(filter.value)}
          >
            {filter.label}
          </button>
        ))}
      </div>
      <input
        id="telemitry-import-input"
        type="file"
        accept="application/json,.json"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <button className="pill secondary" onClick={triggerFilePicker}>
        Import data
      </button>
    </div>
  );
}

export default Toolbar;
