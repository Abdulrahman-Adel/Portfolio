import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { scrollToElement } from '@/lib/utils';

interface NavItem {
  label: string;
  href: string;
}

const NAV_ITEMS: NavItem[] = [
  { label: 'About', href: 'about' },
  { label: 'Experience', href: 'experience' },
  { label: 'Projects', href: 'projects' },
  { label: 'Skills', href: 'skills' },
  { label: 'Publications', href: 'publications' },
];

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleNavClick = (href: string) => {
    scrollToElement(href);
    setMobileMenuOpen(false);
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 bg-white z-50 transition-all ${scrolled ? 'shadow-sm border-b border-gray-200' : ''}`}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <a 
              href="#hero" 
              className="flex-shrink-0 flex items-center"
              onClick={(e) => {
                e.preventDefault();
                handleNavClick('hero');
              }}
            >
              <span className="text-lg font-bold text-primary">Abdulrahman Adel</span>
            </a>
          </div>
          
          {/* Desktop Nav Links */}
          <div className="hidden md:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => (
              <Button
                key={item.href}
                variant="navLink"
                onClick={() => handleNavClick(item.href)}
              >
                {item.label}
              </Button>
            ))}
            <Button
              variant="navButton"
              onClick={() => handleNavClick('contact')}
            >
              Contact
            </Button>
          </div>
          
          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center">
            <button 
              aria-expanded={mobileMenuOpen}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-primary hover:bg-gray-100 transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <i className={`text-xl ${mobileMenuOpen ? 'ri-close-line' : 'ri-menu-line'}`}></i>
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile Menu */}
      <div className={`md:hidden border-t border-gray-200 ${mobileMenuOpen ? 'block' : 'hidden'}`}>
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          {NAV_ITEMS.map((item) => (
            <Button
              key={item.href}
              variant="navLinkMobile"
              className="w-full justify-start"
              onClick={() => handleNavClick(item.href)}
            >
              {item.label}
            </Button>
          ))}
          <Button
            variant="navLinkMobile"
            className="w-full justify-start text-primary"
            onClick={() => handleNavClick('contact')}
          >
            Contact
          </Button>
        </div>
      </div>
    </nav>
  );
}
