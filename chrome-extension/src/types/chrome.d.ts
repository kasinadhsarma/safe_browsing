declare namespace chrome {
  export namespace storage {
    export interface StorageChange {
      newValue?: any;
      oldValue?: any;
    }

    export interface StorageChanges {
      [key: string]: StorageChange;
    }

    export interface StorageArea {
      get(keys: string | string[] | null, callback: (items: { [key: string]: any }) => void): void;
      set(items: { [key: string]: any }, callback?: () => void): void;
    }

    export const local: StorageArea;
    export const sync: StorageArea;
    
    export const onChanged: {
      addListener(callback: (changes: StorageChanges, namespace: string) => void): void;
      removeListener(callback: (changes: StorageChanges, namespace: string) => void): void;
    };
  }
}

interface Window {
  chrome: typeof chrome;
}
