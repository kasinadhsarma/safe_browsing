import { Metadata } from 'next';
import Dashboard from './Dashboard';
// Metadata for the page (optional)
export const metadata: Metadata = {
  title: 'Dashboard - SafeNet',
  description: 'Safe Browsing Dashboard for monitoring and managing web activity.',
};

// Main Page Component
export default function Page() {
  return <Dashboard />;
}
