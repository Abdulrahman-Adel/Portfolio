import { SKILLS, TECH_CATEGORIES } from '@/lib/constants';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';

export default function Skills() {
  return (
    <section id="skills" className="py-16 bg-gray-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Skills & Expertise
        </motion.h2>
        
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Technical Skills Column */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h3 className="text-xl font-semibold mb-6">Technical Skills</h3>
              
              {SKILLS.map((skill, index) => (
                <motion.div 
                  key={skill.name}
                  className="mb-4"
                  initial={{ opacity: 0, width: 0 }}
                  whileInView={{ opacity: 1, width: "100%" }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{skill.name}</span>
                    <span className="text-sm text-gray-700">{skill.proficiency}</span>
                  </div>
                  <div className="skill-progress h-[6px] bg-gray-200 rounded-md overflow-hidden">
                    <Progress value={skill.level} className="h-full rounded-md" />
                  </div>
                </motion.div>
              ))}
            </motion.div>
            
            {/* Programming & Tools Column */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h3 className="text-xl font-semibold mb-6">Programming & Tools</h3>
              
              {TECH_CATEGORIES.map((category, categoryIndex) => (
                <motion.div 
                  key={category.title}
                  className="mb-6"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.1 * categoryIndex }}
                >
                  <h4 className="font-medium mb-3">{category.title}</h4>
                  <div className="flex flex-wrap gap-2">
                    {category.items.map((item) => (
                      <Badge
                        key={item.name}
                        variant={item.primary ? 'default' : 'gray'}
                        className={`inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium ${item.primary ? 'text-white' : 'text-gray-800'}`}
                      >
                        {item.icon && <i className={`${item.icon} mr-1.5`}></i>}
                        {item.name}
                      </Badge>
                    ))}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
