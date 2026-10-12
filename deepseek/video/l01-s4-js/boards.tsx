// 9 段黑板的索引：Q1（01–05）、Q2（06）、Q3（07–08）、答案（09）。
import React from 'react';
import {B01, B02, B03, B04, B05} from './boards-q1';
import {B06, B07, B08, B09} from './boards-q23';

export const BOARDS: Record<string, React.FC> = {
  '01': B01, '02': B02, '03': B03, '04': B04, '05': B05, '06': B06, '07': B07, '08': B08, '09': B09,
};
