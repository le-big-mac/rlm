import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  const logsDir = path.join(process.cwd(), 'public', 'logs');

  try {
    if (!fs.existsSync(logsDir)) {
      return NextResponse.json({ files: [] });
    }

    const files = fs.readdirSync(logsDir)
      .filter(f => f.endsWith('.jsonl'))
      .sort()
      .reverse()
      .slice(0, 20);

    return NextResponse.json({ files });
  } catch {
    return NextResponse.json({ files: [] });
  }
}
