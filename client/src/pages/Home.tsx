import { useEffect } from 'react';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import Hero from '@/components/sections/Hero';
import About from '@/components/sections/About';
import Experience from '@/components/sections/Experience';
import Projects from '@/components/sections/Projects';
import Skills from '@/components/sections/Skills';
import Publications from '@/components/sections/Publications';
import Certifications from '@/components/sections/Certifications';
import Contact from '@/components/sections/Contact';
import { PERSONAL_INFO } from '@/lib/constants';

export default function Home() {
  useEffect(() => {
    // Update page title
    document.title = `${PERSONAL_INFO.name} - ${PERSONAL_INFO.title}`;
    
    // Handle anchor links and smooth scrolling
    const handleHashChange = () => {
      const hash = window.location.hash;
      if (hash) {
        const id = hash.replace('#', '');
        const element = document.getElementById(id);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }
    };

    // Run initially
    handleHashChange();

    // Add event listener
    window.addEventListener('hashchange', handleHashChange);
    
    // Cleanup
    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  return (
    <div className="font-sans bg-light text-dark">
      <Navbar />
      <main>
        <Hero />
        <About />
        <Experience />
        <Projects />
        <Skills />
        <Publications />
        <Certifications />
        <Contact />
      </main>
      <Footer />
      
      {/* Custom CSS for timeline */}
      <style jsx="true">{`
        .skill-progress {
          height: 6px;
          border-radius: 3px;
          background: #E5E7EB;
          position: relative;
          overflow: hidden;
        }
        
        [data-section] {
          scroll-margin-top: 80px;
        }
      `}</style>
    </div>
  );
}
