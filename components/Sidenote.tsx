import { FC, ReactNode } from 'react';
import styles from './Sidenote.module.css';

interface SidenoteProps {
  children: ReactNode;
  content: ReactNode;
  number?: number;
}

const Sidenote: FC<SidenoteProps> = ({ children, content, number }) => {
  return (
    <div className={styles.sidenoteWrapper}>
      <div className={styles.mainContent}>
        {children}
        <sup className={styles.reference}>{number}</sup>
      </div>
      <aside className={styles.sidenote}>
        <span className={styles.sidenoteNumber}>{number}.</span> {content}
      </aside>
    </div>
  );
};

export default Sidenote;