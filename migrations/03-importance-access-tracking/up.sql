ALTER TABLE current_notes ADD COLUMN importance   INTEGER NOT NULL DEFAULT 5;
ALTER TABLE current_notes ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE current_notes ADD COLUMN last_accessed TEXT;
