import { scrollToElement } from '@/lib/utils';
import { SOCIAL_LINKS, PERSONAL_INFO } from '@/lib/constants';

export default function Footer() {
  const handleNavClick = (href: string) => {
    scrollToElement(href);
  };

  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            <div>
              <h3 className="text-lg font-semibold mb-4">Abdulrahman Adel</h3>
              <p className="text-gray-400 mb-4">
                Data Scientist & Machine Learning Engineer specializing in AI-powered solutions and deep learning innovations.
              </p>
              <div className="flex space-x-4">
                {SOCIAL_LINKS.map((link) => (
                  <a 
                    key={link.name}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-white transition-colors"
                    aria-label={link.name}
                  >
                    <i className={`${link.icon} text-xl`}></i>
                  </a>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
              <ul className="space-y-2">
                <li>
                  <button
                    onClick={() => handleNavClick('about')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    About
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => handleNavClick('experience')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Experience
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => handleNavClick('projects')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Projects
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => handleNavClick('skills')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Skills
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => handleNavClick('publications')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Publications
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => handleNavClick('contact')}
                    className="text-gray-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Contact
                  </button>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">Contact</h3>
              <ul className="space-y-2">
                <li className="flex items-center">
                  <i className="ri-mail-line mr-2 text-gray-400"></i>
                  <a 
                    href={`mailto:${PERSONAL_INFO.email}`}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    {PERSONAL_INFO.email}
                  </a>
                </li>
                <li className="flex items-center">
                  <i className="ri-phone-line mr-2 text-gray-400"></i>
                  <span className="text-gray-400">{PERSONAL_INFO.phone.primary}</span>
                </li>
                <li className="flex items-center">
                  <i className="ri-map-pin-line mr-2 text-gray-400"></i>
                  <span className="text-gray-400">{PERSONAL_INFO.location}</span>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="mt-10 pt-6 border-t border-gray-800 text-center text-gray-400">
            <p>&copy; {new Date().getFullYear()} Abdulrahman Adel Ibrahim. All rights reserved.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
